from dads_dsl.cpu_load import CpuLoadController


def test_cpu_load_zero_target_starts_no_workers():
    controller = CpuLoadController(0)
    controller.start()
    try:
        assert controller.stats()["cpu_load_target"] == 0
    finally:
        controller.stop()
