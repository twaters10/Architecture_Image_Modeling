import pandas as pd
import io
csv_data = '/Users/tawate/Documents/Architecture_Image_Modeling/data/house_style_list.csv'
df = pd.read_csv(csv_data, header=None)

# Select the first column and rows 2 to end (ignoring the header)
house_styles = df[0][1:].tolist()

DATA_PATH = '/Users/tawate/.cache/kagglehub/datasets/wwymak/architecture-dataset/versions/1/arcDataset/'
TRAINING_DATA_PATH = '/Users/tawate/.cache/kagglehub/datasets/wwymak/architecture-dataset/versions/1/arcDataset/image_class_dset/'
