from nab.detectors.base import AnomalyDetector



class AlphaDetector(AnomalyDetector):

  def __init__(self, *args, **kwargs):

    super(AlphaDetector, self).__init__(*args, **kwargs)

    self.seed = 42


  def handleRecord(self, inputData):
    """Returns a tuple (anomalyScore).
    The anomalyScore is simply a random value from 0 to 1
    """
    anomalyScore = 0.89
    return (anomalyScore, )


  def initialize(self):
    # random.seed(self.seed)
