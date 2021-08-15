import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from causalnex.structure import StructureModel
from IPython.display import Image
from causalnex.plots import plot_structure, NODE_STYLE, EDGE_STYLE
from causalnex.structure.notears import from_pandas
sm = StructureModel()

data = pd.read_csv('../output/dataset/synthetic1.csv', delimiter=',')
sm = StructureModel()
sm = from_pandas(data)
sm.remove_edges_below_threshold(0.8)
viz = plot_structure(
    sm,
    graph_attributes={"scale": "0.5"},
    all_node_attributes=NODE_STYLE.WEAK,
    all_edge_attributes=EDGE_STYLE.WEAK)
Image(viz.draw(format='png'))