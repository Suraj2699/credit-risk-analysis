import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def visualize_categorical_bars(df):
    fig, axes = plt.subplots(3, 3, figsize=(30, 30))

    axes = axes.flatten()

    # Bar chart for Existing Checking Account Status
    df['Existing-Checking-Account-Status'].value_counts().sort_index().plot(kind='bar', color='lightcoral', edgecolor='black', ax=axes[0])
    axes[0].set_title('Distribution of Existing Checking Account Status')
    axes[0].set_xlabel('Account Status')
    axes[0].set_ylabel('Count')
    axes[0].tick_params(axis='x', rotation=45)

    # Bar chart for Purpose
    df['Purpose'].value_counts().sort_index().plot(kind='bar', color='skyblue', edgecolor='black', ax=axes[1])
    axes[1].set_title('Distribution of Purpose')
    axes[1].set_xlabel('Purpose')
    axes[1].set_ylabel('Count')
    axes[1].tick_params(axis='x', rotation=45)

    df['Property'].value_counts().sort_index().plot(kind='bar', color='lightsteelblue', edgecolor='black', ax=axes[2])
    axes[2].set_title('Distribution of Property')
    axes[2].set_xlabel('Property')
    axes[2].set_ylabel('Count')
    axes[2].tick_params(axis='x', rotation=45)

    df['Housing'].value_counts().sort_index().plot(kind='bar', color='darkorchid', edgecolor='black', ax=axes[3])
    axes[3].set_title('Distribution of Housing')
    axes[3].set_xlabel('Housing')
    axes[3].set_ylabel('Count')
    axes[3].tick_params(axis='x', rotation=45)

    df['Credit-History'].value_counts().sort_index().plot(kind='bar', color='olive', edgecolor='black', ax=axes[4])
    axes[4].set_title('Distribution of Credit History')
    axes[4].set_xlabel('Credit History')
    axes[4].set_ylabel('Count')
    axes[4].tick_params(axis='x', rotation=45)

    df['Savings-Account(Bonds)'].value_counts().sort_index().plot(kind='bar', color='cornflowerblue', edgecolor='black', ax=axes[5])
    axes[5].set_title('Distribution of Savings Account/Bonds')
    axes[5].set_xlabel('Savings Account/Bonds')
    axes[5].set_ylabel('Count')
    axes[5].tick_params(axis='x', rotation=45)

    df['Present-Employment-Since'].value_counts().sort_index().plot(kind='bar', color='cyan', edgecolor='black', ax=axes[6])
    axes[6].set_title('Distribution of Present Employment Status')
    axes[6].set_xlabel('Present Employement Status')
    axes[6].set_ylabel('Count')
    axes[6].tick_params(axis='x', rotation=45)

    df['Personal-Status-Sex'].value_counts().sort_index().plot(kind='bar', color='springgreen', edgecolor='black', ax=axes[7])
    axes[7].set_title('Distribution of Personal Status')
    axes[7].set_xlabel('Personal Status')
    axes[7].set_ylabel('Count')
    axes[7].tick_params(axis='x', rotation=45)

    df['Telephone'].value_counts().sort_index().plot(kind='bar', color='tan', edgecolor='black', ax=axes[8])
    axes[8].set_title('Distribution of Telephone Registration')
    axes[8].set_xlabel('Telephone')
    axes[8].set_ylabel('Count')
    axes[8].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()

def visualize_categorical_distributions(df):
    fig, axes = plt.subplots(3, 3, figsize=(30, 30))

    axes = axes.flatten()

    sns.countplot(x='Job', hue='class', data=df, ax=axes[0])
    axes[0].set_title('Probability of Good Credit by Job')

    sns.countplot(x='Housing', hue='class', data=df, ax=axes[1])
    axes[1].set_title('Probability of Good Credit by Housing')

    sns.countplot(x='Present-Employment-Since', hue='class', data=df, ax=axes[2])
    axes[2].set_title('Probability of Good Credit by Employment Status')

    sns.countplot(x='Purpose', hue='class', data=df, ax=axes[3])
    axes[3].set_title('Probability of Good Credit by Purpose')

    sns.countplot(x='Existing-Checking-Account-Status', hue='class', data=df, ax=axes[4])
    axes[4].set_title('Probability of Good Credit by Existing Checking Account Status')

    sns.countplot(x='Savings-Account(Bonds)', hue='class', data=df, ax=axes[5])
    axes[5].set_title('Probability of Good Credit by Savings Account Status')

    sns.countplot(x='Telephone', hue='class', data=df, ax=axes[6])
    axes[6].set_title('Probability of Good Credit by Telephone Registration')

    sns.countplot(x='Installment-Rate(%)', hue='class', data=df, ax=axes[7])
    axes[7].set_title('Probability of Good Credit by Installment-Rate(%)')

    sns.countplot(x='Foreign-Worker', hue='class', data=df, ax=axes[8])
    axes[8].set_title('Probability of Good Credit by Foreign-Worker')

    plt.tight_layout()
    plt.show()

def visualize_continuous_distributions(df):
    fig, axes = plt.subplots(5, 1, figsize=(10, 30))

    axes = axes.flatten()

    sns.histplot(df['Credit-Amount'], kde=True, ax=axes[0])
    axes[0].set_title(f'Distribution of Credit Amount (Skewness: {df['Credit-Amount'].skew():.2f})')

    sns.histplot(df['Duration(Months)'], kde=True, ax=axes[1])
    axes[1].set_title(f'Distribution of Duration (Months) (Skewness: {df['Duration(Months)'].skew():.2f})')

    sns.histplot(df['Age'], kde=True, ax=axes[2])
    axes[2].set_title(f'Distribution of Age (Skewness: {df['Age'].skew():.2f})')

    sns.histplot(df['Debt_to_Income_Ratio'], kde=True, ax=axes[3])
    axes[3].set_title(f'Distribution of Debt to Income Ratio (Skewness: {df['Debt_to_Income_Ratio'].skew():.2f})')

    sns.histplot(df['Credit_Utilization'], kde=True, ax=axes[4])
    axes[4].set_title(f'Distribution of Credit Utilization (Skewness: {df['Credit_Utilization'].skew():.2f})')

    plt.tight_layout()
    plt.show()

def create_boxplots(df):
    fig, axes = plt.subplots(5, 1, figsize=(8, 30))

    axes = axes.flatten()

    df['Age'].plot(kind='box', color='green', ax=axes[0])
    axes[0].set_title('Boxplot of Age Distribution')

    df['Credit-Amount'].plot(kind='box', color='red', ax=axes[1])
    axes[1].set_title('Boxplot of Credit Amount Distribution')

    df['Duration(Months)'].plot(kind='box', color='blue', ax=axes[2])
    axes[2].set_title('Boxplot of Duration in Months Distribution')

    df['Debt_to_Income_Ratio'].plot(kind='box', color='purple', ax=axes[3])
    axes[3].set_title('Boxplot of Debt to Income Ratio Distribution')

    df['Credit_Utilization'].plot(kind='box', color='orange', ax=axes[4])
    axes[4].set_title('Boxplot of Credit Utilization Distribution')

    plt.tight_layout()

    plt.show()

def property_v_class_crossmap(df):
    propertyvclass = pd.crosstab(df['Property'], df['class'])

    plt.plot(figsize=(10, 6))
    sns.heatmap(propertyvclass, annot=True, cmap='Blues', fmt='d', linewidths=0.5)
    plt.title('Heatmap of Property vs. Class')
    plt.xlabel('Class')
    plt.ylabel('Property')
    plt.show()

def purpose_v_class_crossmap(df):
    purposeVclass = pd.crosstab(df['Purpose'], df['class'])

    plt.figure(figsize=(10, 6))
    sns.heatmap(purposeVclass, annot=True, cmap='Blues', fmt='d', linewidths=0.5)
    plt.title('Heatmap of Purpose vs. Class')
    plt.xlabel('Class')
    plt.ylabel('Purpose')
    plt.show()

def correlation_matrix(df):
    corr = df.corr()
    plt.figure(figsize=(20, 20))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.title('Correlation Matrix')

    plt.tight_layout()
    plt.show()

def model_comparison(dictionary):
    """
    Compare the performance of different models using a bar plot.

    Parameters:
    - dictionary: A dictionary containing model names as keys and their respective scores as ROC AUC values.
    """
    plt.figure(figsize=(16, 8))
    sns.barplot(x=list(dictionary.keys()), y=list(dictionary.values()), palette='viridis')
    plt.title('Model Comparison')
    plt.xlabel('Model')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    plt.show()