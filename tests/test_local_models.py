import asyncio
import threading

import pytest

from gujusub.local_models import LocalModels, LocalSpec


class Warm:
    def __init__(self):
        self.warmed = 0

    def warmup(self):
        self.warmed += 1


class Registry(Warm):
    def __init__(self, langs):
        super().__init__()
        self.langs = tuple(langs)

    def __contains__(self, lang):
        return lang in self.langs


class Recorder:
    def __init__(self, gate=None, fail=False):
        self.calls = []
        self.gate = gate
        self.fail = fail

    def engines(self, langs, device, gu_model="full"):
        self.calls.append(("engines", tuple(langs), device, gu_model))
        if self.gate is not None:
            assert self.gate.wait(5)
        if self.fail:
            raise OSError("no model files")
        return Registry(langs)

    def translator(self):
        self.calls.append(("translator",))
        return Warm()

    def lid(self):
        self.calls.append(("lid",))
        return Warm()

    def models(self, spec=None):
        return LocalModels(spec or LocalSpec(), load_engines=self.engines,
                           make_translator=self.translator, load_language_id=self.lid)


def test_sync_load_loads_everything_once_in_order():
    rec = Recorder()
    lm = rec.models(LocalSpec(langs=("gu", "en"), device="mps", gu_model="fast"))
    assert lm.state == "idle"
    lm.load()
    assert rec.calls == [("engines", ("gu", "en"), "mps", "fast"), ("translator",), ("lid",)]
    assert lm.state == "ready"
    assert lm.engines.warmed == 1 and lm.translator.warmed == 1 and lm.language_id.warmed == 1


def test_sync_load_skips_lid_without_en_and_translator_when_off():
    rec = Recorder()
    lm = rec.models(LocalSpec(langs=("gu",), translate=False)).load()
    assert rec.calls == [("engines", ("gu",), "cpu", "full")]
    assert lm.translator is None and lm.language_id is None


def test_sync_load_skips_translator_when_one_exists():
    rec = Recorder()
    lm = rec.models(LocalSpec(langs=("gu",))).load(need_translator=False)
    assert ("translator",) not in rec.calls and lm.translator is None


def test_sync_load_failure_sets_error_and_raises():
    lm = Recorder(fail=True).models()
    with pytest.raises(OSError):
        lm.load()
    assert lm.state == "error" and "no model files" in lm.error


def test_background_load_runs_once_and_reports_on_the_loop():
    gate = threading.Event()
    rec = Recorder(gate=gate)
    lm = rec.models(LocalSpec(langs=("gu",)))
    done = []

    async def run():
        loop = asyncio.get_running_loop()

        def on_done(models):
            assert threading.current_thread() is threading.main_thread()
            done.append(models.state)

        assert lm.start_background(loop, True, on_done) is True
        assert lm.start_background(loop, True, on_done) is False  # already loading
        assert lm.state == "loading" and lm.engines is None
        waiter = asyncio.create_task(lm.wait())
        await asyncio.sleep(0.02)
        assert not waiter.done()
        gate.set()
        assert await asyncio.wait_for(waiter, 2) is True
        assert lm.start_background(loop, True, on_done) is False  # ready: never again

    asyncio.run(run())
    assert done == ["ready"]
    assert [c[0] for c in rec.calls] == ["engines", "translator"]
    assert lm.engines.langs == ("gu",)


def test_background_load_failure_is_reported():
    lm = Recorder(fail=True).models()
    done = []

    async def run():
        lm.start_background(asyncio.get_running_loop(), False, lambda m: done.append(m.error))
        return await asyncio.wait_for(lm.wait(), 2)

    assert asyncio.run(run()) is False
    assert lm.state == "error" and done == ["no model files"]


def test_wait_without_a_load_returns_readiness():
    lm = Recorder().models()

    async def run():
        return await lm.wait()

    assert asyncio.run(run()) is False
    lm.load()
    assert asyncio.run(run()) is True


def test_on_done_failure_still_releases_waiters():
    lm = Recorder().models(LocalSpec(langs=("gu",), translate=False))

    def bad(_models):
        raise RuntimeError("callback bug")

    async def run():
        lm.start_background(asyncio.get_running_loop(), False, bad)
        return await asyncio.wait_for(lm.wait(), 2)

    assert asyncio.run(run()) is True


def test_translator_only_job_loads_once_without_engines():
    rec = Recorder()
    lm = rec.models()
    done = []

    async def run():
        loop = asyncio.get_running_loop()
        assert lm.start_translator(loop, lambda m: done.append(m.translator_state))
        assert not lm.start_translator(loop, lambda m: done.append("again"))
        await _until(lambda: done)

    asyncio.run(run())
    assert done == ["ready"] and rec.calls == [("translator",)]
    assert lm.translator.warmed == 1 and lm.state == "idle"


def test_translator_only_job_failure_is_reported():
    lm = LocalModels(LocalSpec(), load_engines=None, load_language_id=None,
                     make_translator=lambda: (_ for _ in ()).throw(OSError("no ct2")))
    done = []

    async def run():
        lm.start_translator(asyncio.get_running_loop(), lambda m: done.append(m))
        await _until(lambda: done)

    asyncio.run(run())
    assert lm.translator_state == "error" and lm.translator_error == "no ct2"


async def _until(cond):
    for _ in range(400):
        if cond():
            return
        await asyncio.sleep(0.005)
    raise AssertionError("condition not reached")
