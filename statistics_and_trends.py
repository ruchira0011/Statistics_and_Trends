import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as ss
import seaborn as sns


def plot_relational_plot(df):
    '''
    Generating a scatter plot to visualize the relationship between
      math score and reading score, grouped by gender.
    - Useing Seaborn's scatterplot to show correlation.
    - Colors the points based on gender.
    - Saves the plot as 'relational_plot.png'.
    '''
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        data=df,
        x="math score",
        y="reading score",
        hue="gender",
        alpha=0.7,
        ax=ax)
    ax.set_title("Math Score vs Reading Score by Gender")
    plt.savefig('relational_plot.png')
    return


def plot_categorical_plot(df):
    '''
    Plots a box plot to visualize the distribution of writing scores
    based on parental education level and gender.
    - Uses Seaborn's boxplot to show the distribution of writing scores.
    - Groups the data by parental education level and gender.
    - Rotates x-axis labels for better readability.
    - Then saving the plot as 'categorical_plot.png'.'
    '''
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.boxplot(data=df,
                x="parental level of education",
                y="writing score",
                hue="gender",
                ax=ax)
    ax.set_title("Writing Score by Parental Education Level and Gender")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('categorical_plot.png')
    return


def plot_statistical_plot(df):
    """
    Plotting a pair plot for numerical variables in the dataset
     to visualize pairwise relationships between numerical variables.
    - Uses Seaborn's pairplot with corner=True to create a corner plot.
    - Then saving the plot as 'statistical_plot.png'.
    """
    # Create the pair plot
    g = sns.pairplot(df, corner=True)

    # Adjust the figure size for better title placement
    g.figure.set_size_inches(10, 10)

    # Set the title and adjust its position
    g.figure.suptitle("Pairwise Relationships of Numerical Variables", y=1.05)

    # Save the plot to a file
    g.figure.savefig('statistical_plot.png')

    # Close the plot
    plt.close()
    return


def statistical_analysis(df, col: str):
    '''
    Performs statistical analysis on a specified column of the dataset.
    Calculates the mean, standard deviation, skewness, and excess kurtosis.
    '''
    mean = df[col].mean()
    stddev = df[col].std()
    skewness = ss.skew(df[col])
    excess_kurtosis = ss.kurtosis(df[col])
    return mean, stddev, skewness, excess_kurtosis


def preprocessing(df):
    '''
    Preprocesses the dataset by performing initial exploratory data analysis.
    - Displays basic statistics for the dataset.
    - Shows the first and last few rows of the dataset.
    - Computes and displays the correlation matrix for numerical columns.
    '''
    # Display basic statistics
    print("Dataset Description:\n", df.describe())

    # Show first and last few rows to get an overview of the data
    print("\nFirst few rows:\n", df.head())
    print("\nLast few rows:\n", df.tail())

    # Compute correlation matrix (only for numerical columns)
    numeric_cols = df.select_dtypes(include=['number'])
    correlation_matrix = numeric_cols.corr()
    print("\nCorrelation Matrix:\n", correlation_matrix)

    return df


def writing(moments, col):
    """
    Prints and interprets the statistical moments
    (mean, standard deviation, skewness, and excess kurtosis)
    - Displays the calculated values of the moments.
    - Interprets skewness and kurtosis to describe the shape
      of the data distribution.

    Parameters:
    - moments (tuple): A tuple containing the mean, standard deviation,
      skewness, and excess kurtosis.
    - col (str): The name of the column being analyzed.
    """
    print(f'For the attribute {col}:')
    print(f'Mean = {moments[0]:.2f}, '
          f'Standard Deviation = {moments[1]:.2f}, '
          f'Skewness = {moments[2]:.2f}, and '
          f'Excess Kurtosis = {moments[3]:.2f}.')
    # Not skewed and mesokurtic can be defined with asymmetries <-2 or >2.
    if abs(moments[2]) > 2:
        skew_type = "right-skewed" if moments[2] > 0 else "left-skewed"
    else:
        skew_type = "not skewed"

    if moments[3] < 0:
        kurtosis_type = "platykurtic"
    elif moments[3] == 0:
        kurtosis_type = "mesokurtic"
    else:
        kurtosis_type = "leptokurtic"

    print(f'The data was {skew_type} and {kurtosis_type}.')
    return


def main():
    # Read the data from a CSV file and load it into a DataFrame
    df = pd.read_csv('data.csv')

    # Perform preprocessing on the data
    df = preprocessing(df)

    # Setting the analysing column name for the variable
    col = 'math score'

    '''
    Plotting relational plots to visualize relationships
     between variables in the data
    '''
    plot_relational_plot(df)

    # Plotting statistical plots for analysis of the data distribution
    plot_statistical_plot(df)

    '''
    Plotting categorical plots to examine relationships between
      categorical variables
    '''
    plot_categorical_plot(df)

    # Perform statistical analysis on the specified column
    moments = statistical_analysis(df, col)

    # Write the analysis results to an output
    writing(moments, col)

    return


if __name__ == '__main__':
    main()
