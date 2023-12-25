from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SequentialFeatureSelector as sfs
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
from io import BytesIO
import base64

app = Flask(__name__, template_folder="templates")


def create_visualizations(data):
    plt.figure(figsize=(12, 6))

    # Create a boxplot
    plt.subplot(2, 1, 1)
    sns.boxplot(data)
    plt.title('Boxplot of Data')
    plt.xlabel('Features')
    plt.ylabel('Values')

    # Save the boxplot to a BytesIO object
    boxplot_stream = BytesIO()
    plt.savefig(boxplot_stream, format='png')
    boxplot_stream.seek(0)

    # Encode the boxplot to base64 for HTML rendering
    encoded_boxplot = base64.b64encode(boxplot_stream.read()).decode('utf-8')
    plt.close()  # Close the plot to free up resources

    # Create a correlation heatmap
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 2)
    correlation_matrix = data.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap')

    # Save the correlation heatmap to a BytesIO object
    heatmap_stream = BytesIO()
    plt.savefig(heatmap_stream, format='png')
    heatmap_stream.seek(0)

    # Encode the correlation heatmap to base64 for HTML rendering
    encoded_heatmap = base64.b64encode(heatmap_stream.read()).decode('utf-8')
    plt.close()  # Close the plot to free up resources

    # Return the images as HTML img tags
    boxplot_image = f'<img src="data:image/png;base64,{encoded_boxplot}" alt="Boxplot">'
    correlation_heatmap = f'<img src="data:image/png;base64,{encoded_heatmap}" alt="Correlation Heatmap">'

    return boxplot_image, correlation_heatmap




def linear_regression(data, Target_variable):
    result = "<p>Exploratory Data Analysis (EDA)</p><br>"

    # Create visualizations (boxplot and correlation heatmap)
    boxplot_image, correlation_heatmap = create_visualizations(data)
    result += f'{boxplot_image}<br>{correlation_heatmap}<br>'

    x = list(data.columns)
    for i in x:
        if data[i].dtypes == 'object':
            data.drop(i, axis=1, inplace=True)



    x = list(data.columns)
    ## Treating outliers
    
    for i in x:
        q1 = data[i].quantile(0.25)
        q3 = data[i].quantile(0.75)
        iqr = q3 - q1
        ul = q3 + 1.5 * iqr
        ll = q1 - 1.5 * iqr
        if (data[i].any() < ll or data[i].any() > ul):
            data[i].clip(upper=ul, lower=ll, inplace=True)

    ## Filling null values
    result += "<br><br>Missing values:-"
    for i in x:
        if (data[i].isnull().sum() / len(data[i]) * 100) >= 40:
            data.drop(i, axis=1, inplace=True)
        elif data[i].isna().sum() > 0:
            result += f"missing values in {i} is {data[i].isna().sum()}<br>"
        else:
            result += "<br>There are no missing values in the columns<br>"
            break

    for i in x:
        if data[i].dtypes in ("float64", "int64"):
            q1 = data[i].quantile(0.25)
            q3 = data[i].quantile(0.75)
            iqr = q3 - q1
            upperL = q3 + 1.5 * iqr
            lowerL = q1 - 1.5 * iqr
            if data[i].min() < lowerL or data[i].max() > upperL:
                data[i].fillna(data[i].median(), inplace=True)
            else:
                data[i].fillna(data[i].mean(), inplace=True)
        else:
            data[i].fillna(data[i].mode()[0], inplace=True)

    # Handling missing values in the target variable
    data.dropna(subset=[Target_variable], inplace=True)

    Y = pd.DataFrame(data[Target_variable])
    if Y.isna().sum().any() > 0:
        Y.fillna(Y.mean(), inplace=True)

    X = data.drop(Target_variable, axis=1)

    ## Combining the dependent and independent variable
    df = pd.concat([X, Y], axis=1)

    ## Checking skewness
    s = float(Y.skew())

    if s != 0:
        Y = np.log(Y)
        ## Model building
        result += "<br><br>Train dataset contains 70% of the dataset and Test Dataset contains 30% of the dataset<br>"
        X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.7, random_state=134)
        train = pd.concat([y_train, X_train], axis=1)
        lreg = LinearRegression()

        # Feature selection
        Model = sfs(lreg, n_features_to_select=5, direction='backward', scoring='r2', cv=5)
        Model.fit(X_train, y_train)
        f_out = list(Model.get_feature_names_out().flatten())

        X_train = X_train.loc[:, f_out]
        train = pd.concat([y_train, X_train], axis=1)
        Model = LinearRegression()
        Model.fit(X_train, y_train)

        ## Prediction
        train_target = train.iloc[:, 0]
        train['fitted_value'] = np.round(Model.predict(X_train), 2)
        train['Residual'] = np.round(train_target - train.fitted_value, 2)

        r2 = r2_score(train_target, train.fitted_value)
        r2_train = (np.round(r2, 2) * 100)

        # Calculate the mean squared error
        model_mse = mean_squared_error(train_target, train['fitted_value'])
        # Calculate the root mean squared error
        model_rmse = np.sqrt(model_mse)
        rmse_train = (np.exp(model_rmse))

        # Prediction on test data
        X_test = X_test.loc[:, f_out]
        test = pd.concat([y_test, X_test], axis=1)

        test_target = test.iloc[:, 0]
        test['Prediction'] = np.round(Model.predict(X_test), 2)
        test['Residual'] = np.round(test_target - test.Prediction, 2)

        r2 = r2_score(test_target, test.Prediction)
        r2_test = (np.round(r2, 2) * 100)

        # Calculate the mean squared error
        model_mse = mean_squared_error(test_target, test['Prediction'])
        # Calculate the root mean squared error
        model_rmse = np.sqrt(model_mse)
        rmse_test = np.exp(model_rmse)

        Y = np.sqrt(Y)
        ## Model building
        X2_train, X2_test, y2_train, y2_test = train_test_split(X, Y, train_size=0.7, random_state=134)
        train2 = pd.concat([y2_train, X2_train], axis=1)
        lreg = LinearRegression()

        # Feature selection
        Model2 = sfs(lreg, n_features_to_select=5, direction='backward', scoring='r2', cv=5)
        Model2.fit(X2_train, y2_train)
        f_out2 = list(Model2.get_feature_names_out().flatten())

        X2_train = X2_train.loc[:, f_out2]
        train2 = pd.concat([y2_train, X2_train], axis=1)
        Model2 = LinearRegression()
        Model2.fit(X_train, y_train)

        ## Prediction
        train2_target = train.iloc[:, 0]
        train2['fitted_value'] = np.round(Model.predict(X2_train), 2)
        train2['Residual'] = np.round(train2_target - train2.fitted_value, 2)

        r2_2 = r2_score(train2_target, train2.fitted_value)
        r2_2_train = (np.round(r2_2, 2) * 100)

        # Calculate the mean squared error
        model_mse = mean_squared_error(train2_target, train2['fitted_value'])
        # Calculate the root mean squared error
        model_rmse2_train = np.sqrt(model_mse)
        rmse2_train = np.square(model_rmse2_train)

        # Prediction on test data
        X2_test = X2_test.loc[:, f_out]
        test2 = pd.concat([y2_test, X2_test], axis=1)

        test2_target = test.iloc[:, 0]
        test2['Prediction'] = np.round(Model.predict(X2_test), 2)
        test2['Residual'] = np.round(test2_target - test2.Prediction, 2)

        r2_2 = r2_score(test_target, test.Prediction)
        r2_2_test = (np.round(r2_2, 2) * 100)

        # Calculate the mean squared error
        model_mse = mean_squared_error(test_target, test['Prediction'])
        # Calculate the root mean squared error
        model_rmse2 = np.sqrt(model_mse)
        rmse_2_test = np.square(model_rmse2)

        if r2_train > r2_2_train:
            result += "<br>Features selected from the independent variable for analysis are: " + str(f_out)
            result += '<br>R2 score for model Performance on Train : ' + str(r2_train)
            result += "<br>RMSE of Train Data : " + str(rmse_train)
            result += '<br>R2 score for model Performance on Test : ' + str(r2_test)
            result += "<br>RMSE of Test Data : " + str(rmse_test)
        else:
            result += "<br>Features selected from the independent variable for analysis are: " + str(f_out2)
            result += '<br>R2 score for model Performance on Train : ' + str(r2_2_train)
            result += "<br>RMSE of Train Data : " + str(rmse2_train)
            result += '<br>R2 score for model Performance on Test : ' + str(r2_2_test)
            result += "<br>RMSE of Test Data : " + str(rmse_2_test)
    else:
        ## Model building
        X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.7, random_state=134)
        train = pd.concat([y_train, X_train], axis=1)
        lreg = LinearRegression()

        # Feature selection
        Model = sfs(lreg, n_features_to_select=5, direction='backward', scoring='r2', cv=5)
        Model.fit(X_train, y_train)
        f_out = list(Model.get_feature_names_out().flatten())
        X_train = X_train.loc[:, f_out]
        train = pd.concat([y_train, X_train], axis=1)
        Model = LinearRegression()
        Model.fit(X_train, y_train)

        ## Prediction
        train_target = train.iloc[:, 0]
        train['fitted_value'] = np.round(Model.predict(X_train), 2)
        train['Residual'] = np.round(train_target - train.fitted_value, 2)

        r2 = r2_score(train_target, train.fitted_value)
        result += '<br>R2 score for model Performance on Train : ' + str(np.round(r2, 2) * 100)

        # Calculate the mean squared error
        model_mse = mean_squared_error(train_target, train['fitted_value'])
        # Calculate the root mean squared error
        model_rmse = np.sqrt(model_mse)

        result += "<br>RMSE of Train Data : " + str(np.round(model_rmse))

        # Prediction on test data
        X_test = X_test.loc[:, f_out]
        test = pd.concat([y_test, X_test], axis=1)

        test_target = test.iloc[:, 0]
        test['Prediction'] = np.round(Model.predict(X_test), 2)
        test['Residual'] = np.round(test_target - test.Prediction, 2)

        r2 = r2_score(test_target, test.Prediction)
        result += '<br>R2 score for model Performance on Test : ' + str(np.round(r2, 2) * 100)

        # Calculate the mean squared error
        model_mse = mean_squared_error(test_target, test['Prediction'])
        # Calculate the root mean squared error
        model_rmse = np.sqrt(model_mse)
        result += "<br>RMSE of Test Data : " + str(np.round(model_rmse, 2))
    result = result.replace('\n', '<br>')
    return result


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the uploaded dataset
        dataset = request.files['dataset']

        # Read the dataset into a DataFrame
        data = pd.read_csv(dataset)  # Assuming CSV format, modify as needed

        # Get the target variable from the form
        target_variable = request.form['target_variable']

        # Call the linear_regression function and capture the result
        result = linear_regression(data, target_variable)

        # Add any additional processing or rendering logic here

        # Return the result to be displayed in the template
        return render_template('index.html', result=result)

    return render_template('index.html')


# Add a new route to handle the form submission
@app.route('/process_data', methods=['POST'])
def process_data():
    # Get the uploaded dataset
    dataset = request.files['dataset']

    # Read the dataset into a DataFrame
    data = pd.read_csv(dataset)  # Assuming CSV format, modify as needed

    # Get the target variable from the form
    target_variable = request.form['target_variable']

    # Call the linear_regression function and capture the result
    result = linear_regression(data, target_variable)

    # Return the result to be displayed in the template
    return render_template('index.html', result=result)


if __name__ == '__main__':
    app.run(debug=True)
