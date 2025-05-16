import traceback
from . import read
from . import DEC
from . import MapReducer
import Adaptive_EBat_DBN.DBN
import CBF_DBN.DBN
import Hybrid_NN.Hybrid_NN
import SSPO_DQN.DQN
import WOA_BRNN.brnn

def callmain(dts, tr):  # dataset name, training percentage
    A, Tpr, Tnr = [], [], []

    try:
        # ------------------- Phase 1: Read Dataset -------------------
        data, label = read.read_input(dts)
        if data is None or label is None:
            raise ValueError("Data or Label is empty. Check dataset path or format.")

        # ------------------- Phase 2: DEC Clustering -------------------
        x, y = DEC.main(data, label)

        # ------------------- Phase 3: MapReduce -------------------
        Data, Target = MapReducer.Map_Reducer(x, y, dts, tr, A, Tpr, Tnr)

        # ------------------- Phase 4: Run All Classifiers -------------------
        SSPO_DQN.DQN.cal_metrics(Data, Target, tr, A, Tpr, Tnr)
        Adaptive_EBat_DBN.DBN.classify(Data, Target, tr, A, Tpr, Tnr)
        CBF_DBN.DBN.classify(Data, Target, tr, A, Tpr, Tnr)
        WOA_BRNN.brnn.classify(Data, Target, tr, A, Tpr, Tnr)

        # ------------------- Phase 5: Final Hybrid Classifier -------------------
        ACC, TPR, TNR = Hybrid_NN.Hybrid_NN.classify(Data, Target, tr, A, Tpr, Tnr)

        return ACC, TPR, TNR

    except Exception as e:
        print("❌ Error occurred in callmain:")
        traceback.print_exc()
        return [0] * 6, [0] * 6, [0] * 6  # dummy values on failure
