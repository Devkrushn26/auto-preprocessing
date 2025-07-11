import streamlit as st
import pandas as pd
import io
from ydata_profiling import ProfileReport
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler,StandardScaler

from sklearn.base import BaseEstimator, TransformerMixin

class LogTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns  # List of columns to apply log transform

    def fit(self, X, y=None):
        return self  # No training needed

    def transform(self, X):
        X = X.copy()
        for col in self.columns:
            if col in X.columns:
                X[col] = np.log1p(X[col])  # log1p(x) = log(x + 1) to handle zero values
        return X

 ## Histograms
def plot_histograms(df):
     numeric_df=df.select_dtypes(include=['number'])
     fig, ax =plt.subplots(figsize=(12,6))
     numeric_df.hist(ax=ax, bins=30)
     return fig

## count plot
def plot_countplots(df):
    categorical_cols = df.select_dtypes(include=['object']).columns
    figs = []
    for col in categorical_cols:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.countplot(data=df, x=col, ax=ax)
        ax.set_title(f"Count Plot of {col}")
        figs.append(fig)
    return figs

## Box plot
def plot_boxplots(df):
    numeric_cols = df.select_dtypes(include=['number']).columns
    figs = []
    for col in numeric_cols:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.boxplot(data=df, x=col, ax=ax)
        ax.set_title(f"Box Plot of {col}")
        figs.append(fig)
    return figs

#### feachers eng fug

# Finding duplicate rows
def find_duplicate_rows(df):
    duplicate_rows = df[df.duplicated(keep=False)]
    duplicate_count = df.duplicated().sum()
    duplicate_indices = df[df.duplicated()].index.tolist()
    return duplicate_count, duplicate_indices

zz=0
def get_unique_name(base_name="tnx"):
    global zz
    zz=zz+1
    return f"{base_name}_{zz}"



st.sidebar.title("Navigation")
selected_section = st.sidebar.radio("Go to", ["EDA", "Fechers engnering", "Model Bulding"])

df=pd.DataFrame()

if selected_section=="EDA":

    st.title("CSV file Upload")

    uploaded_file=st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        df=pd.read_csv(uploaded_file,sep=None, engine="python")
        st.session_state["df"]=df
        

    if st.button("Uplord csv"):
       
             st.write("data preview")
             st.dataframe(df.head())
             #data shape
             st.write("Data shape ",df.shape)
              #data ingfo
             st.write(" Data Info:")
             buffer = io.StringIO()
             df.info(buf=buffer)
             info_str = buffer.getvalue()
             st.text_area("Data Info:", info_str, height=200)
             st.success("uplorded")
    if uploaded_file is not None:

        st.write("### Genret EDA repot for this dataset")
        if st.button("Genreate EDA repot"):
            profile = ProfileReport(df, explorative=True)
            profile_html = profile.to_html()
            st.write("### Data Analysis Report")
            components.html(profile_html, height=800, scrolling=True)
        

        # EDA

        st.subheader("Click BElow to Explore the Dataset")

        with st.expander("Show Column Names"):
            st.write(f"Columns :{list(df.columns)}")

        with st.expander("Show Missing Values"):
            st.write(df.isnull().sum())

        with st.expander("Show Summary Statistics"):
            st.write(df.describe())
        
        with st.expander("Show Feature Distributions (Histogram)"):
            st.pyplot(plot_histograms(df))

        with st.expander("Show Count Plots for Categorical Features"):
            for fig in plot_countplots(df):
                st.pyplot(fig)

        with st.expander("Show Box Plots for Outlier Detection"):
            for fig in plot_boxplots(df):
                st.pyplot(fig)






#feachers eng

elif selected_section=="Fechers engnering":

    st.subheader("Click Below to do fechers engnering")

    if st.session_state["df"] is not None:
        df = st.session_state["df"]  # Retrieve df from session state

        with st.expander("Handle Duplicate Rows"):
            duplicate_count, duplicate_indices = find_duplicate_rows(df)
            st.write(f"Duplicate Count: {duplicate_count}")
            st.write(f"Duplicate Indices: {duplicate_indices}")

            option = st.radio("Choose an option:", ["Keep Duplicate Rows", "Remove Duplicate Rows"])

            if st.button("Apply Changes"):
                if option == "Remove Duplicate Rows":
                    df = df.drop_duplicates()  # Remove duplicates
                    st.session_state["df"] = df  # Save changes
                    st.success("Duplicate rows removed successfully!")
                else:
                    st.info("No changes made. Keeping duplicate rows.")
        
        ##Drop unnecessary columns
        with st.expander("Drop unnecessary columns"):
            option=st.radio("Choose an columns :",list(df.columns))
            if st.button("Drop"):
                df=df.drop([option],axis=1)
                st.session_state["df"]=df
                st.success("Drop successfully")

        ##Handel Missing Valuses
        with st.expander("Handel Missing Valuses"):
            st.write(df.isnull().sum())
            option=st.radio("Select method :",["Remove Rows with Missing Values","Remove Columns with Too Many Missing Values","Fill with a Constant Value","Fill with Mean","Fill with Median","Fill with Mode"])
            
            ##Remove Rows with Missing Values
            if option=="Remove Rows with Missing Values":
                if st.button("Remove rows",key="remove_row"):
                    df = df.dropna()  # Fix: Do not use inplace=True
                    st.session_state["df"] = df  # Update session state
                    st.success("Rows with missing values dropped successfully!")

            #Remove Columns with Too Many Missing Values
            elif option=="Remove Columns with Too Many Missing Values":
                col=st.selectbox("Choose an columns :",list(df.columns))
                if st.button("Drop",key="Drop_col"):
                    df = df.drop(columns=[col])
                    st.session_state["df"] = df  
                    st.success(f"Column '{col}' dropped successfully!")

            #Fill with a Constant Value
            elif option=="Fill with a Constant Value":
                col=st.selectbox("Fill with a Constant Value :",["Replaces NaNs with 0","Replaces missing categorical values with Unkown"])
                if col=="Replaces NaNs with 0":
                    if st.button("Replaces",key="fill_with_zero"):
                        #df.fillna(0, inplace=True)
                        numerical_cols = df.select_dtypes(include=["number"]).columns.tolist()
                        for i in numerical_cols:
                            df[i].fillna(0,inplace=True)
                        st.session_state["df"] = df  
                        st.success("Replaces NaNs with 0 successfully!")
                        
                elif col=="Replaces missing categorical values with Unkown":
                    if st.button("Replaces", key="fill_with_ukonwn"):
                        #df['Category'].fillna("Unknown", inplace=True) 
                        categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
                        for i in categorical_cols:
                            df[i].fillna("Unknown", inplace=True)
                        st.session_state["df"] = df
                        st.success("Replaces missing categorical values with Unkown successfully! ")


            #Fill with Mean
            
            elif option=="Fill with Mean":
                if st.button("Replace with Mean",key="Repalce_with_Mean"):
                    numerical_cols = df.select_dtypes(include=["number"]).columns.tolist()
                    for i in numerical_cols:
                        df[i].fillna(df[i].mean(),inplace=True)
                    st.session_state["df"] = df
                    st.success("Replaces missing values with Mean successfully! ")

            #Fill with Median
            elif option=="Fill with Median":
                if st.button("Replace with Median",key="Repalce_with_Median"):
                    numerical_cols = df.select_dtypes(include=["number"]).columns.tolist()
                    for i in numerical_cols:
                        df[i].fillna(df[i].median(),inplace=True)
                    st.session_state["df"] = df
                    st.success("Replaces missing values with Median successfully! ")
                
            #Fill with Mode
            elif option=="Fill with Mode":
                if st.button("Replace with Mode",key="Repalce_with_Mode"):
                    for i in list(df.columns):
                        df[i].fillna(df[i].mode()[0],inplace=True)
                    st.session_state["df"] = df
                    st.success("Replaces missing values with Mode successfully! ")  

        Pipeline=Pipeline(steps=[])
        st.session_state["Pipeline"] = Pipeline

        #Handal outlietrs
        with st.expander("Handl Outliers"):
            numerical_cols = df.select_dtypes(include=["number"]).columns.tolist()
            st.write("Outliers count each colums")
            Lower_bound=[]
            Uper_bound=[]
            count_outlier=[]
            col_out_name=[]

            for i in numerical_cols:
                Q1=df[i].quantile(0.25)
                Q2=df[i].quantile(0.75)
                IQR=Q2-Q1
                lower=Q1-1.5*IQR
                upper=Q2+1.5*IQR
                Lower_bound.append(lower)
                Uper_bound.append(upper)
                count=df[(df[i]<lower)|(df[i]>upper)].shape[0]
                count_outlier.append(count)
                if count>0:
                    col_out_name.append(i)
                st.write(f"{i} : {count}")
            
           
            out_cal=st.radio("Select column to handal outlier",col_out_name) 
            out_math=st.selectbox("Select method for handal outlier : ",["Remove Raw","Replace with boundary value","Log Transformation"])   
            if st.button("Handal",key="outlier_handal"):

                #remove raw 
                if out_math=="Remove Raw":
                    df=df[(df[out_cal]>=Lower_bound[numerical_cols.index(out_cal)])&(df[out_cal]<=Uper_bound[numerical_cols.index(out_cal)])]
                    st.session_state["df"]=df
                    st.success(f"outliers removed in columm {out_cal}")

                if out_math=="Replace with boundary value":
                    df[out_cal]=np.where(df[out_cal]<Lower_bound[numerical_cols.index(out_cal)],Lower_bound[numerical_cols.index(out_cal)],df[out_cal])
                    df[out_cal]=np.where(df[out_cal]>Uper_bound[numerical_cols.index(out_cal)],Uper_bound[numerical_cols.index(out_cal)],df[out_cal])
                    st.session_state["df"]=df
                    st.success("Replaced with Boundary Value")

                if out_math=="Log Transformation":
                    Pipeline=st.session_state.get["Pipeline"]                                                           #####################################################
                    Pipeline.steps.append((get_unique_name("log_transformation"),LogTransformer(columns=out_cal)))
                    df[out_cal]=LogTransformer(columns=out_cal)
                    st.session_state["Pipeline"]=Pipeline                                                                                                                    #### Pipeline
                   
                    st.session_state["df"]=df
                    st.success(f"{out_cal} transformed in log ")
            
#       # chose data type
        if "model_type_is_categarical" not in st.session_state:
            st.session_state["model_type_is_categarical"] = False

        model_type_is_categarical=st.session_state.get("model_type_is_categarical")
        with st.expander("Chose data type "):
            data_type=st.radio("Chose data type",["Regression Data","Classification Data"])
            if st.button("Submit"):
                if "data_type" not in st.session_state:
                    st.session_state["data_type"] = None
                ##### data_type
                if data_type=="Classification Data":
                    model_type_is_categarical=True
                    st.session_state["model_type_is_categarical"]=model_type_is_categarical
                st.success(f"set data type {data_type} sucsesfully")


        #chose target colum
        x=pd.DataFrame()
        y=pd.DataFrame()                                                 #####  x , y ,target
        with st.expander("Chose Feaches and Target columns"):
            a=df.copy()
            target =st.radio("Chose target column",list(a.columns))
            if st.button("Target Submit"):
                st.session_state["target"]=target
                y=df[target]
                x=a.drop([target],axis=1)
                st.session_state["x"]=x
                st.session_state["y"]=y
                st.session_state["column_names"] = x.columns.tolist()                  ### x colume name
                st.success("Target set sucsesfully!")

        #one hot and label encoding 

        if "x" in st.session_state:
            x = st.session_state["x"]
        
            col_a=x.select_dtypes(include=["object"]).columns.tolist()
 
            with st.expander("Categorical Encoding"):
                if col_a:  # Only show if there are categorical columns
                    cal_b = st.radio("Choose column for Categorical Encoding", col_a)
                    encoder = st.selectbox("Select Encoder", ["One-Hot Encoder", "Label Encoder"])

                    if st.button("Encode", key="handle_categorical"):
                        # Convert back to DataFrame if needed
                        if not isinstance(x, pd.DataFrame):
                            x = pd.DataFrame(x, columns=st.session_state["column_names"])

                        if encoder == "One-Hot Encoder":
                            trans = ColumnTransformer(
                                transformers=[(f"ohe_{cal_b}", OneHotEncoder(sparse=False), [cal_b])],
                                remainder="passthrough"
                            )
                            x_arry = trans.fit_transform(x)
                            ohe_feature_names = trans.named_transformers_[f"ohe_{cal_b}"].get_feature_names_out([cal_b])
                            #passthrough_cols = [col for col in x.columns if col != cal_b]
                            #new_column_names = list(ohe_feature_names) + passthrough_cols
                            new_column_names=trans.get_feature_names_out()
                            x = pd.DataFrame(x_arry, columns=new_column_names)
                            col_a=x.select_dtypes(include=["object"]).columns.tolist()
                            #x = pd.DataFrame(x_arry, columns=st.session_state["column_names"])

                            st.session_state["column_names"] = new_column_names
                            Pipeline=st.session_state.get["Pipeline"]
                            Pipeline.steps.append((f"ohe_{cal_b}", trans))
                            st.success("One-Hot Encoding Done")

                        elif encoder == "Label Encoder":
                            trans = ColumnTransformer(
                                transformers=[("label_encoder", OrdinalEncoder(), [cal_b])],
                                remainder="passthrough"
                            )
                            x = trans.fit_transform(x)
                            st.session_state["column_names"] = st.session_state["column_names"]
                            if not isinstance(x, pd.DataFrame):
                                x = pd.DataFrame(x, columns=st.session_state["column_names"])
                            # No new column names for OrdinalEncoder
                              # keep same
                            Pipeline=st.session_state.get["Pipeline"]
                            Pipeline.steps.append((f"labelen_{cal_b}", trans))
                            st.success("Label Encoding Done")

                        # Update session state
                        st.session_state["x"] = x
                        st.session_state["Pipeline"] = Pipeline
                else:
                    st.warning("No categorical columns found to encode.")            
                                
        # target veriable encoding
        
        with st.expander("If you want to Encode your target wariable"):
            model_type_is_categarical=st.session_state.get("model_type_is_categarical")
            y=st.session_state.get("y")
            if model_type_is_categarical and y is not None and not y.empty:
                encoder = st.selectbox("Select Encoder", ["One-Hot Encoder", "Label Encoder"],key="target_encoder")
                if st.button("Encode", key="handgle_categorical"):

                        y = y.dropna().reset_index(drop=True)


                        # Convert back to DataFrame if needed


                        if encoder == "One-Hot Encoder":
                            ohe = OneHotEncoder(sparse=False)
                            y = pd.DataFrame(y)
                            y_encoded = ohe.fit_transform(y[[target]])
                            new_columns = ohe.get_feature_names_out([y.columns[0]])
                            y = pd.DataFrame(y_encoded, columns=new_columns)
                            st.success("One-Hot Encoding Done")
                            

                            
                            #ohe_feature_names = trans.named_transformers_[f"ohe_{cal_b}"].get_feature_names_out([cal_b])
                            #passthrough_cols = [col for col in x.columns if col != cal_b]
                            
                            
                          
                            

                           

                        elif encoder == "Label Encoder":
                            y = pd.DataFrame(y)
                            trans = ColumnTransformer(
                                transformers=[("label_encoder", OrdinalEncoder(), [y.columns[0]])],
                                remainder="passthrough"
                            )
                            y_arry = trans.fit_transform(y)
                            if not isinstance(y, pd.DataFrame):
                                y = pd.DataFrame(y_arry, columns=st.session_state["column_names"])
                            
                            st.success("Label Encoding Done")

                        # Update session state
                        st.session_state["y"] = y

        st.write("data preview")
        st.dataframe(x.head())

        x=st.session_state.get("x")
        Pipeline=st.session_state.get("Pipeline")
        st.subheader("Normalization")
        norm = st.radio("Select Normalizer", ["Min-Max", "Standardization(Z-score)"])
        if st.button("Normalize", key="normalize"):
                if norm == "Min-Max":
                    min_max = MinMaxScaler()
                    norm_data = min_max.fit_transform(x)
                    x = pd.DataFrame(norm_data, columns=x.columns)
                    st.session_state["x"] = x
                    Pipeline.steps.append(('min-max', min_max))
                    st.session_state["Pipeline"] = Pipeline
                    st.success("Min-Max Scaler Done")


                elif norm == "Standardization(Z-score)":
                    scaler = StandardScaler()
                    norm_data = scaler.fit_transform(x)
                    x = pd.DataFrame(norm_data, columns=x.columns)
                    st.session_state["x"] = x
                    Pipeline.steps.append(('standard-scaler', scaler))
                    st.session_state["Pipeline"] = Pipeline
                    st.success("Standard Scaler Done")
        
        
        x=st.session_state.get("x")
        y=st.session_state.get("y")
        data=pd.concat([x,y],axis="columns")
        st.write("data preview")
        st.dataframe(data.head())
        CSV = data.to_csv(index=False).encode('utf-8')
        st.download_button(label="Downlord data as Csv",data=CSV,file_name="sample_data.csv",mime='text/csv')


elif selected_section=="Model Bulding":
    x=st.session_state.get("x")
    y=st.session_state.get("y")
    model_type_is_categarical=st.session_state.get("model_type_is_categarical")

    if model_type_is_categarical==False:
        algo=["Linear Regression","Polynomial Regression","Support Vector Regression (SVR)","Decision Tree Regressor","Random Forest Regressor","K-Nearest Neighbors (KNN) Regressor"]
        st.radio("Select Regression Algorithm",algo)





        
                            



                        

            









        

        

        


            




                
                        

            







    




    



