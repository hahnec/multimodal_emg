from tests.test_multimodal_dtype import MultimodalTester
from tests.test_multimodal_emg import EchoMultiModalTester
from tests.test_batch_multimodal_emg import BatchEchoMultiModalTester
from tests.test_peak_detect import PeakDetectTester
from tests.test_picmus_data import TestPicmusData


if __name__ == '__main__':

    test_classes = [
        MultimodalTester,
        EchoMultiModalTester,
        BatchEchoMultiModalTester,
        TestPicmusData,
                    ]

    for test_class in test_classes:

        # instantiate test object
        obj = test_class()
        obj.setUp()

        # switch off plots for headless
        obj.plot_opt = False

        obj.main()

        del obj

    print('\nRan through all tests')
