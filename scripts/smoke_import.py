from radar_dnn_mcts.models import AutoregressiveDecoder, BatchDecoder, LatentDynamics, RadarSchedulerModel


def main() -> None:
    core = RadarSchedulerModel()
    AutoregressiveDecoder(core)
    BatchDecoder()
    LatentDynamics()
    print("smoke_import ok")
    print("models=reencode,muzero,autoregressive,batch")


if __name__ == "__main__":
    main()
