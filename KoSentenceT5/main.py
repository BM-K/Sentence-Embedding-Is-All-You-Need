from model.setting import Setting, Arguments
from model.simcse.processor import Processor


def main(args, logger) -> None:
    processor = Processor(args)
    config = processor.model_setting()
    logger.info('Model Setting Complete')

    if args.train == 'True':
        logger.info('Start Training')

        for epoch in range(args.epochs):

            processor.train(epoch+1)

    if args.test == 'True':
        logger.info("Start Test")

        processor.test()

        processor.metric.print_size_of_model(config['model'])
        processor.metric.count_parameters(config['model'])


if __name__ == '__main__':
    args, logger = Setting().run()
    main(args, logger)
