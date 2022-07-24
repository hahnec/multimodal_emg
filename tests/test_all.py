from tests.test_multimodal_dtype import MultimodalTester
from tests.test_multimodal_emg import EchoMultiModalTester

test_classes = [
    MultimodalTester,
    EchoMultiModalTester,
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
