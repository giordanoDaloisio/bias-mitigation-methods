from aif360.algorithms.preprocessing import Reweighing, DisparateImpactRemover
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from utility import *
plt.rc('font', family='serif')
sns.set_theme()

n_samples = 10000
n_features = 10
n_informative = 4
n_sensitive = 2

data = pd.read_csv('synthetic/synthetic.csv')
data.head()

unprivileged_group = [{'s': 0}]
privileged_group = [{'s': 1}]

bias_data = data.copy()
bias_data.loc[(bias_data['s'] == 0) & (bias_data.index < int(n_samples/2)) , '10'] = 0.0
bias_data_bin = BinaryLabelDataset(df = bias_data, label_names=['10'], protected_attribute_names=['s'])
sample_bias_data = sample_dataset(bias_data.copy(),
 [(bias_data['s']==1), (bias_data['s']==0)], 
 bias_data['10']==1, bias_data['10']==0, ['s'], '10')
plt.show()