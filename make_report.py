from src.utils import load_data
import sweetviz as sv

data_dir = './input'
df, train, test = load_data(data_dir)

feature_config = sv.FeatureConfig(skip=["id", "is_train"])
my_report = sv.compare([train, "Training Data"], [test, "Test Data"], "Global_Sales", feature_config)

my_report.show_html()