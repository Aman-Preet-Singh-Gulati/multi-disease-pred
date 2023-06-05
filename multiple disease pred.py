import pickle
import streamlit as st
from streamlit_option_menu import option_menu

from util import classify
from keras.models import load_model
from PIL import Image
import numpy as np
import pandas as pd


st.set_page_config(
   page_title="Multiple Disease Prediction",
#    page_icon='./Logo/SRA_Logo.ico',
)

# loading the saved models
diabetes_model = pickle.load(open('ml_model/diabetes_prediction.pkl', 'rb'))
heart_disease_model = pickle.load(open('ml_model/heart-disease-prediction-knn-model.pkl','rb'))
# kidney_disease_model = pickle.load(open('ml_model/kidney_disease_prediction.pkl','rb'))


# sidebar for navigation
with st.sidebar:
    selected = option_menu('Multiple Disease Prediction System',          
                          ['Diabetes Prediction',
                           'Heart Disease Prediction',
                           'Pneumonia Prediction'],
                          icons=['activity','heart','file-medical'],
                          #menu_icon = "hospital",
                          default_index=0)
    
# Diabetes Prediction Page
if (selected == 'Diabetes Prediction'):

    selected_diabetes_menu = option_menu(          
                        menu_title = "Diabetes Prediction using ML", # required
                        options = ["Home", "Predict", "Feedback"], # required
                        icons = ["house", "book", "envelope"],
                        menu_icon = "cast",
                        orientation ="horizontal",
                        key="1",
                        default_index=0)

    if (selected_diabetes_menu == 'Home'):
        # Introduction
        st.markdown(
            """
            With over **463 million** adults affected globally in **2019**, and the numbers projected to reach 700 million by 2045, 
            diabetes has become a pressing epidemic. But here's the good news: up to **90%** of type **2 diabetes cases** are preventable ❕
            """
        )

        # Heading
        st.markdown("### Harnessing Machine Learning to Combat Diabetes 📋")

        # Short points
        st.markdown("- **Early Risk Identification:** Predicts diabetes risk using genetic, health, and lifestyle data. Up to 90% of type 2 diabetes cases can be prevented. [Source: CDC](https://www.cdc.gov/diabetes/basics/prevention.html)")
        st.markdown("- **Personalized Treatment Optimization:** Tailors treatment plans for improved diabetes management based on individual data. [Source: Journal of Diabetes Science and Technology](https://journals.sagepub.com/doi/10.1177/1932296819849876)")
        st.markdown("- **Continuous Glucose Monitoring (CGM):** Predicts blood glucose levels using real-time CGM data, enabling proactive interventions. [Source: Diabetes Technology & Therapeutics](https://www.liebertpub.com/doi/full/10.1089/dia.2019.0137)")
        st.markdown("- **Remote Monitoring and Telemedicine:** Facilitates effective diabetes management from a distance using machine learning-powered remote monitoring and telemedicine. [Source: Journal of Diabetes Science and Technology](https://journals.sagepub.com/doi/full/10.1177/1932296818816491)")
        
        st.markdown("---")

        # Read the CSV file
        df = pd.read_csv('dataset/diabetes.csv')

        # Display the first 10 rows of the table
        st.markdown("### Sample dataset :open_file_folder:")
        st.table(df.head(7).reset_index(drop=True))     

    if (selected_diabetes_menu == 'Predict'):
        # page title
        # st.title('Diabetes Prediction using ML')
        
        # getting the input data from the user
        col1, col2, col3 = st.columns(3)
        
        with col1:
            Pregnancies = st.text_input('Number of Pregnancies')    
        with col2:
            Glucose = st.text_input('Glucose Level')
        with col3:
            BloodPressure = st.text_input('Blood Pressure value')
        with col1:
            SkinThickness = st.text_input('Skin Thickness value')
        with col2:
            Insulin = st.text_input('Insulin Level')
        with col3:
            BMI = st.text_input('BMI value')
        with col1:
            DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
        with col2:
            Age = st.text_input('Age of the Person')
        
        # code for Prediction
        diab_diagnosis = ''

        # creating a button for Prediction
        
        if st.button('Diabetes Test Result'):
            diab_prediction = diabetes_model.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
            
            if (diab_prediction[0] == 1):
                diab_diagnosis = 'The person is diabetic'
            else:
                diab_diagnosis = 'The person is not diabetic'
            
        st.success(diab_diagnosis)
    
    if(selected_diabetes_menu == 'Feedback'):

        with st.form(key='columns_in_form2',clear_on_submit=True): #set clear_on_submit=True so that the form will be reset/cleared once it's submitted
            #st.write('Please help us improve!')
            Name=st.text_input(label='Enter Your Name') #Collect user feedback
            Email=st.text_input(label='Enter Your Email') #Collect user feedback
            Message=st.text_input(label='Enter Your Message') #Collect user feedback
            submitted = st.form_submit_button('Submit')
            if submitted:
                st.write('Thanks for your contacting us. We will respond to your questions or inquiries as soon as possible!')


# Heart Disease Prediction Page
if (selected == 'Heart Disease Prediction'): 

    selected_heart_menu = option_menu(          
                        menu_title = "Heart Disease Prediction using ML", # required
                        options = ["Home", "Predict", "Feedback"], # required
                        icons = ["house", "book", "envelope"],
                        menu_icon = "cast",
                        orientation ="horizontal",
                        key="2",
                        default_index=0) 

    if (selected_heart_menu == 'Home'):
        # Introduction
        st.markdown(
            """
            With over **17.9 million** deaths worldwide in **2019**, heart disease remains a leading cause of mortality. 
            Machine learning offers promising solutions to combat this global health challenge ❕
            """
        )

        # Heading
        st.markdown("### Harnessing Machine Learning for Heart Disease Classification 📋")

        # Short points
        st.markdown("- **Early Detection:** Machine learning models analyze various patient data, such as medical records, symptoms, and test results, to detect early signs of heart disease. Early detection can lead to timely interventions and improved patient outcomes. [Source: American Heart Association](https://www.heart.org/en/health-topics/heart-attack/diagnosing-a-heart-attack)")
        st.markdown("- **Risk Assessment:** Machine learning algorithms can assess an individual's risk of developing heart disease by analyzing factors like age, blood pressure, cholesterol levels, and lifestyle habits. This allows for personalized prevention strategies and targeted interventions. [Source: Journal of the American College of Cardiology](https://www.sciencedirect.com/science/article/pii/S0735109714000198)")
        st.markdown("- **Prediction and Diagnosis:** Machine learning models can predict the likelihood of heart disease based on patient characteristics and aid in accurate diagnosis. This assists healthcare professionals in making informed decisions about treatment plans. [Source: European Society of Cardiology](https://academic.oup.com/eurheartj/article/39/7/508/4095049)")
        st.markdown("- **Treatment Optimization:** Machine learning can optimize treatment plans by analyzing patient data, treatment responses, and medical literature. This helps healthcare providers tailor interventions and medications to maximize efficacy and improve patient outcomes. [Source: Frontiers in Cardiovascular Medicine](https://www.frontiersin.org/articles/10.3389/fcvm.2021.686482/full)")

        st.markdown("---")

        # Read the CSV file
        df = pd.read_csv('dataset/heart.csv')

        # Display the first 10 rows of the table
        st.markdown("### Sample dataset :open_file_folder:")
        st.table(df.head(7).reset_index(drop=True))     

    if (selected_heart_menu == 'Predict'):            
        # page title
        # st.title('Heart Disease Prediction using ML')
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.text_input('Age')
        with col2:
            sex = st.text_input('Sex')
        with col3:
            cp = st.text_input('Chest Pain types')
        with col1:
            trestbps = st.text_input('Resting Blood Pressure')
        with col2:
            chol = st.text_input('Serum Cholestoral in mg/dl')
        with col3:
            fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')
        with col1:
            restecg = st.text_input('Resting Electrocardiographic results')
        with col2:
            thalach = st.text_input('Maximum Heart Rate achieved')
        with col3:
            exang = st.text_input('Exercise Induced Angina')
        with col1:
            oldpeak = st.text_input('ST depression induced by exercise')
        with col2:
            slope = st.text_input('Slope of the peak exercise ST segment')
        with col3:
            ca = st.text_input('Major vessels colored by flourosopy')
        with col1:
            thal = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')
        
        # code for Prediction
        heart_diagnosis = ''
        
        # creating a button for Prediction
        
        if st.button('Heart Disease Test Result'):
            heart_prediction = heart_disease_model.predict([[age, sex, cp, trestbps, chol, fbs, restecg,thalach,exang,oldpeak,slope,ca,thal]])                          
            
            if (heart_prediction[0] == 1):
                heart_diagnosis = 'The person is having heart disease'
            else:
                heart_diagnosis = 'The person does not have any heart disease'
            
        st.success(heart_diagnosis)

    if(selected_heart_menu == 'Feedback'):

        with st.form(key='columns_in_form2',clear_on_submit=True): #set clear_on_submit=True so that the form will be reset/cleared once it's submitted
            #st.write('Please help us improve!')
            Name=st.text_input(label='Enter Your Name') #Collect user feedback
            Email=st.text_input(label='Enter Your Email') #Collect user feedback
            Message=st.text_input(label='Enter Your Message') #Collect user feedback
            submitted = st.form_submit_button('Submit')
            if submitted:
                st.write('Thanks for your contacting us. We will respond to your questions or inquiries as soon as possible!')



# Covid-19 disease Prediction Page
if (selected == 'Pneumonia Prediction'):

    selected_pneumonia_menu = option_menu(          
                        menu_title = "Pneumonia Prediction using Deep learning", # required
                        options = ["Home", "Predict", "Feedback"], # required
                        icons = ["house", "book", "envelope"],
                        menu_icon = "cast",
                        orientation ="horizontal",
                        key="3",
                        default_index=0) 

    if (selected_pneumonia_menu == 'Home'):
        # Introduction
        st.markdown(
            """
            Pneumonia is a significant global health concern, causing millions of deaths each year. In fact, pneumonia is responsible for approximately 15% of all deaths in children under the age of 5 worldwide. Deep learning techniques offer powerful tools to aid in the prediction and diagnosis of pneumonia, improving patient outcomes. 
            """
        )

        # Heading
        st.markdown("### Deep Learning for Pneumonia Prediction 📋")

        # Short points
        st.markdown("- **Automated Diagnosis:** Deep learning models can analyze chest X-ray images with high accuracy, assisting in automated diagnosis of pneumonia. This can help healthcare professionals save time and make more informed treatment decisions. [Source: IEEE Access](https://ieeexplore.ieee.org/document/8979143)")
        st.markdown("- **Improved Sensitivity and Specificity:** Deep learning algorithms can enhance the sensitivity and specificity of pneumonia detection, reducing the chances of misdiagnosis. This leads to improved patient care and appropriate treatment plans. [Source: Radiology](https://pubs.rsna.org/doi/full/10.1148/radiol.2018180958)")
        st.markdown("- **Early Detection:** Deep learning models can identify subtle patterns and features in chest X-rays that may indicate early stages of pneumonia. Early detection allows for prompt intervention and treatment, potentially reducing disease progression. [Source: Nature Communications](https://www.nature.com/articles/s41467-019-09257-w)")
        st.markdown("- **Efficient Triage:** Deep learning algorithms can prioritize chest X-ray scans, assisting in the efficient triage of patients. By prioritizing severe cases, healthcare providers can allocate resources effectively and ensure timely treatment for those in critical condition. [Source: Journal of Digital Imaging](https://link.springer.com/article/10.1007/s10278-019-00236-3)")

        st.markdown("---")

        # Display the first 10 rows of the table
        st.markdown("### Sample dataset :open_file_folder:")

        # Load the images
        normal_image_1 = Image.open(r"dataset\chest_xray\train\NORMAL\IM-0115-0001.jpeg")
        normal_image_2 = Image.open(r"dataset\chest_xray\train\NORMAL\IM-0117-0001.jpeg")
        pneumonia_image_1 = Image.open(r"dataset\chest_xray\train\PNEUMONIA\person1_bacteria_1.jpeg")
        pneumonia_image_2 = Image.open(r"dataset\chest_xray\train\PNEUMONIA\person1_bacteria_2.jpeg")

        # Display the images with equal width
        col1, col2 = st.columns(2)
        with col1:
            st.image(normal_image_1, use_column_width=True, caption="Normal Image 1")
            st.image(normal_image_2, use_column_width=True, caption="Normal Image 2")
        with col2:
            st.image(pneumonia_image_1, use_column_width=True, caption="Pneumonia Image 1")
            st.image(pneumonia_image_2, use_column_width=True, caption="Pneumonia Image 2")


    if (selected_pneumonia_menu == 'Predict'):
        # page title
        # st.title('Pneumonia Prediction using Deep learning')

        # upload file
        file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])

        # load classifier
        model = load_model('./deep_learning_model/pneumonia_classifier.h5')

        # load class names
        with open('./deep_learning_model/labels.txt', 'r') as f:
            class_names = [a[:-1].split(' ')[1] for a in f.readlines()]
            f.close()

        # display image
        if file is not None:
            image = Image.open(file).convert('RGB')
            st.image(image, use_column_width=True)

            # classify image
            class_name, conf_score = classify(image, model, class_names)

            # write classification
            st.write("## {}".format(class_name))
            st.write("### score: {}%".format(int(conf_score * 1000) / 10))
    
    if(selected_pneumonia_menu == 'Feedback'):

        with st.form(key='columns_in_form2',clear_on_submit=True): #set clear_on_submit=True so that the form will be reset/cleared once it's submitted
            #st.write('Please help us improve!')
            Name=st.text_input(label='Enter Your Name') #Collect user feedback
            Email=st.text_input(label='Enter Your Email') #Collect user feedback
            Message=st.text_input(label='Enter Your Message') #Collect user feedback
            submitted = st.form_submit_button('Submit')
            if submitted:
                st.write('Thanks for your contacting us. We will respond to your questions or inquiries as soon as possible!')


