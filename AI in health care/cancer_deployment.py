import streamlit as st
import pandas as pd
import numpy as np
import sklearn
import pickle
model=pickle.load(open('cancer_prediction.pkl','rb'))

nav=st.sidebar.radio('Navigation',['Home','Cancer','Kidney','Diabetes','Liver','Heart'])
if nav=='Home':


    st.title('Health Care App')
    st.image('heath_care_ai.jpg',width=600,use_column_width=True)

    st.write(' This is a Machine Learning Based health care app.'
             'The prediction about the disease is made with the help of the machine learning model which itself is trained on the thousands of the datasets'
             )
    st.header('Details')
    st.write('Welcome to the home page of the app.')
    st.header('Overview')
    st.write('In todays time we see a lot of the shortage of the doctors in the world especially in India.A lot of people are suffering a lot without the help of the proper medical checkup.Also most of the cases many cases arise leading to dealth due to lack of timely medical checkup')

    st.write('So to cope up with all of those problems this app is designed which would prove its benefits upto much extent.')

    st.header('Application')
    st.write('* To remove the dependencies on the doctors')
    st.write('* To help out the poor and helpless people with the normal medical checkup')
    st.write('* To help people avoid paying huge amount to the doctors unnecessarily')
    st.write('* To extend the role of the technology in the medical field')

    st.write('This is mainly based on as the application of the machine learning,meant to be employed in the remote and the downtrodden area.')
    st.header('This App can be used to predict the Following Disease:'  )

    st.subheader('1. Breast Cancer')
    st.subheader('2. Kidney Disease')
    st.subheader('3. Diabetes')
    st.subheader('4. Liver Problem')
    st.subheader('5. Heart Disease')

    st.header('Created By:-')

    st.write('TEJPAL KUMAWAT')
    st.write('INDIAN INSTITUTE OF TECHNOLOGY BOMBAY')
    st.write('26-DEC-2020')



if nav=='Cancer':
    st.title('1.Breast Cancer  ')
    st.image('breast.jpg',width=550)
    st.write('Breast cancer is cancer that develops from breast tissue. Signs of breast cancer may include a lump in the breast, a change in breast shape, dimpling of the skin, fluid coming from the nipple, a newly inverted nipple, or a red or scaly patch of skin.')
    st.header('Symptoms')
    st.write('Signs and symptoms of breast cancer may include:'
             )
    st.write('* A breast lump or thickening that feels different from the surrounding tissue')
    st.write('* Change in the size, shape or appearance of a breast')
    st.write('* Changes to the skin over the breast, such as dimpling')
    st.write('* A newly inverted nipple')
    st.write('* Peeling, scaling, crusting or flaking of the pigmented area of skin surrounding the nipple (areola) or breast skin')
    st.write('* Redness or pitting of the skin over your breast, like the skin of an orange')

    st.header('Causes')
    st.write('Doctors know that breast cancer occurs when some breast cells begin to grow abnormally. These cells divide more rapidly than healthy cells do and continue to accumulate, forming a lump or mass. Cells may spread (metastasize) through your breast to your lymph nodes or to other parts of your body.'
              'Breast cancer most often begins with cells in the milk-producing ducts (invasive ductal carcinoma). Breast cancer may also begin in the glandular tissue called lobules (invasive lobular carcinoma) or in other cells or tissue within the breast.'
              "Researchers have identified hormonal, lifestyle and environmental factors that may increase your risk of breast cancer. But its not clear why some people who have no risk factors develop cancer, yet other people with risk factors never do. It's likely that breast cancer is caused by a complex interaction of your genetic makeup and your environment")

    st.header('Predict Breast Cancer Patient here')
    st.subheader('Please the fill patient medical data here :')
    a=st.number_input('Radius_mean')
    b=st.number_input('texture_mean')
    c=st.number_input('perimeter_mean')
    d=st.number_input('area_mean')
    e=st.number_input('smoothness_mean')
    f=st.number_input('compactness_mean')
    g=st.number_input('concavity_mean')
    h=st.number_input('concave points_mean')
    i=st.number_input('symmetry_mean')
    j=st.number_input('fractal_dimension_mean	')
    k=st.number_input('radius_se')
    l=st.number_input('texture_se')
    m=st.number_input('perimeter_se')
    n=st.number_input('area_se')
    o=st.number_input('smoothness_se')
    p=st.number_input('compactness_se')
    q=st.number_input('concavity_se')
    r=st.number_input('concave points_se	')
    s=st.number_input('symmetry_se')
    t=st.number_input('fractal_dimension_se')
    u=st.number_input('radius_worst')
    v=st.number_input('texture_worst')
    w=st.number_input('perimeter_worst')
    x=st.number_input('area_worst')
    y=st.number_input('smoothness_worst')
    z=st.number_input('compactness_worst')
    aa=st.number_input('concavity_worst')
    bb=st.number_input('concave points_worst')
    cc=st.number_input('symmetry_worst')
    dd=st.number_input('fractal_dimension_worst')
    input=[[a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z,aa,bb,cc,dd]]
    prediction = model.predict_proba(input)[:,1]
    pred=model.predict(input)
    output = np.round(prediction, 2)
    if st.button('predict'):

        if pred==1:
            st.write('your tumor is Malignant')
            st.write('Probability of having Cancer to the patient is '+str(output))
            st.write('Please Consult to the doctor as early as possible')
        else:
            st.write('your tumor is Benign')
            st.write('Probability of having cancer to the patient is'+str(output))
            st.write('You are safe , but keep doing medical checkup of ourself')
model2=pickle.load(open('kidney_prediction.pkl','rb'))
if nav=='Kidney':
    st.title('2. Kidney Disease Prediction')
    st.image('kidney.jpeg',width=550)

    st.header('Overview')
    st.write("The kidneys play key roles in body function, not only by filtering the blood and getting rid of waste products, but also by balancing the electrolyte levels in the body, controlling blood pressure, and stimulating the production of red blood cells.")
    st.write("The kidneys are located in the abdomen toward the back, normally one on each side of the spine. They get their blood supply through the renal arteries directly from the aorta and send blood back to the heart via the renal veins to the vena cava. (The term 'renal' is derived from the Latin name for kidney.")

    st.write('* Lethargy')
    st.write('* Weakness')
    st.write('* Shortness of breath')
    st.write('* Generalized swelling (edema)')
    st.write('* Generalized weakness due to anemia')
    st.write('* Loss of appetite')
    st.write('* Fatigue')
    st.write('* Congestive heart failure')
    st.write('* Metabolic acidosis ')
    st.write('* High blood potassium (hyperkalemia)')
    st.write('* Fatal heart rhythm disturbances (arrhythmias) including ventricular tachycardia and ventricular fibrillation')
    st.write('* Rising urea levels in the blood (uremia) may lead to brain encephalopathy, pericarditis (inflammation of the heart lining), or low calcium blood levels (hypocalcemia)')

    st.header('Predict Kidney Disease of  Patient here')
    st.subheader('Please the fill patient medical data here :')

    def user_data_input():
       # ['age', 'bgr', 'bu', 'sc', 'sod', 'pcv', 'wc', 'sg_random',
         #'hemo_random', 'dm_random', 'cad_random']
        a=st.number_input('Age')
        b=st.number_input('bgr')
        c=st.number_input('bu')
        d=st.number_input('sc')
        e=st.number_input('sod')
        f=st.number_input('pcv')
        g=st.number_input('wc')
        h=st.number_input('sg')
        i=st.number_input('hemo')
        j=st.selectbox('dm',['Yes','No'],index=0)


        k = st.selectbox('cad', ['Yes', 'No'], index=0)
        dataframe=pd.DataFrame({'a':a,
                                'b':b,
                                'c':c,
                                'd':d,
                                'e':e,
                                'f':f,
                                'g':g,
                                'h':h,
                                'i':i,
                                'j':j,
                                'k':k},index=[0])
        return dataframe
    input=user_data_input()

    input['j']=input['j'].replace('Yes',1).replace('No',0)
    input['k']=input['k'].replace('Yes',1).replace("No",0)


    prediction = model2.predict_proba(input)
    pred = model2.predict(input)
    output = np.round(prediction, 2)
    if st.button('predict'):

        st.write('probability of kidney Disease is',output)
model3=pickle.load(open('liver_prediction (1).pkl','rb'))
if nav=='Liver':
    st.title('Liver Disease Prediction')
    st.image('liver.jpg',width=550)
    st.header('Overview')
    st.write("Liver disease is any disturbance of liver function that causes illness. The liver is responsible for many critical functions within the body and should it become diseased or injured, the loss of those functions can cause significant damage to the body. Liver disease is also referred to as hepatic disease.")

    st.write("Liver disease is a broad term that covers all the potential problems that cause the liver to fail to perform its designated functions. Usually, more than 75% or three quarters of liver tissue needs to be affected before a decrease in function occurs.")
    st.header('Symptons')
    st.write('* nausea')
    st.write('* vomiting')
    st.write('* right upper quadrant abdominal pain, and')
    st.write('* jaundice (a yellow discoloration of the skin due to elevated bilirubin concentrations in the bloodstream')
    st.header('Predict Liver Disease of  Patient here')
    st.subheader('Please the fill patient medical data here :')
    def user_data_input():
       # ['age', 'bgr', 'bu', 'sc', 'sod', 'pcv', 'wc', 'sg_random',
         #'hemo_random', 'dm_random', 'cad_random']
        a=st.number_input('Age')
        b=st.selectbox('gender',['male','female'],index=0)
        c=st.number_input('Total_Bilirubin	')
        d=st.number_input('Direct_Bilirubin')
        e=st.number_input('Alkaline_Phosphotase')
        f=st.number_input('Alamine_Aminotransferase')
        g=st.number_input('	Aspartate_Aminotransferase')
        h=st.number_input('Total_Protiens	')
        i=st.number_input('Albumin')
        j=st.number_input('Albumin_and_Globulin_Ratio')




        dataframe=pd.DataFrame({'a':a,
                                'b':b,
                                'c':c,
                                'd':d,
                                'e':e,
                                'f':f,
                                'g':g,
                                'h':h,
                                'i':i,
                                'j':j,
                                },index=[0])
        return dataframe
    input=user_data_input()
    input['b']=input['b'].replace('male',1).replace('female',0)

    prediction = model3.predict_proba(input)
    pred = model3.predict(input)
    output = np.round(prediction, 4)
    if st.button('predict'):

        st.write('probability of liver Disease is',output)
model4=pickle.load(open('diabetes_prediction.pkl','rb'))
if nav=='Diabetes':
    st.title('Diabetes Disease Prediction')
    st.image('diabetes.jpg', width=550)
    st.header('Overview')
    st.write('Diabetes is a disease that occurs when your blood glucose, also called blood sugar, is too high. Blood glucose is your main source of energy and comes from the food you eat. Insulin, a hormone made by the pancreas, helps glucose from food get into your cells to be used for energy.')
    st.header('Symptoms')
    st.write('* heart disease')
    st.write('* stroke')
    st.write('* kidney disease')
    st.write('* eye problems')
    st.write('* dental disease')
    st.write('nerve damage')
    st.header('Predict Diabetes Disease of  Patient here')
    st.subheader('Please the fill patient medical data here :')


    def user_data_input():
        # ['age', 'bgr', 'bu', 'sc', 'sod', 'pcv', 'wc', 'sg_random',
        # 'hemo_random', 'dm_random', 'cad_random']
        a = st.number_input('Pregencies')
        #b = st.selectbox('gender', ['male', 'female'], index=0)
        c = st.number_input('Glucose	')
        d = st.number_input('BloodPressure')
        e = st.number_input('SkinThickness')
        f = st.number_input('Insulin')
        g = st.number_input('BMI')
        h=st.number_input('DiabetesPedigreeFunction')
        i=st.number_input('Age')

        dataframe=pd.DataFrame({'a':a,

                                'c':c,
                                'd':d,
                                'e':e,
                                'f':f,
                                'g':g,
                                'h':h,
                                'i':i,

                                },index=[0])
        return dataframe
    input=user_data_input()
    prediction = model4.predict_proba(input)
    pred = model4.predict(input)
    output = np.round(prediction, 4)
    if st.button('predict'):
        st.write('probability of Diabetes Disease is', output)

model5=pickle.load(open('heart_prediction.pkl','rb'))
if nav=='Heart':
    st.title('Heart Disease Prediction')
    st.image('heart.jpg', width=550)
    st.header('Overview')
    st.write("Heart disease describes a range of conditions that affect your heart. Diseases under the heart disease umbrella include blood vessel diseases, such as coronary artery disease; heart rhythm problems (arrhythmias); and heart defects you're born with (congenital heart defects), among others.")
    st.write("The term 'heart disease' is often used interchangeably with the term 'cardiovascular disease.' Cardiovascular disease generally refers to conditions that involve narrowed or blocked blood vessels that can lead to a heart attack, chest pain (angina) or stroke. Other heart conditions, such as those that affect your heart's muscle, valves or rhythm, also are considered forms of heart disease.")
    st.write("Many forms of heart disease can be prevented or treated with healthy lifestyle choices.")
    st.header('Symptoms')
    st.write('* Chest pain, chest tightness, chest pressure and chest discomfort (angina)')
    st.write('* Shortness of breath')
    st.write('* Pain, numbness, weakness or coldness in your legs or arms if the blood vessels in those parts of your body are narrowed')
    st.write('* Pain in the neck, jaw, throat, upper abdomen or back')
    def user_data_input():
        # ['age', 'bgr', 'bu', 'sc', 'sod', 'pcv', 'wc', 'sg_random',
        # 'hemo_random', 'dm_random', 'cad_random']
        a = st.number_input('age')
        b = st.selectbox('sex', ['male', 'female'], index=0)
        c = st.number_input('cp	')
        d = st.number_input('trestbps')
        e = st.number_input('chol')
        f = st.number_input('fbs')
        g = st.number_input('restecg	')
        h=st.number_input('thalach')
        i=st.number_input('exang')
        j=st.number_input('oldpeak')
        k= st.number_input('slope	')
        l= st.number_input('ca')
        m=st.number_input('thal')

        dataframe=pd.DataFrame({'a':a,
                                'b':b,

                                'c':c,
                                'd':d,
                                'e':e,
                                'f':f,
                                'g':g,
                                'h':h,
                                'i':i,
                                'j':j,
                                'k':k,
                                'l':l,
                                'm':m

                                },index=[0])
        return dataframe
    input=user_data_input()
    input['b'] = input['b'].replace('male', 1).replace('female', 0)
    prediction = model5.predict_proba(input)
    pred = model5.predict(input)
    output = np.round(prediction, 4)
    if st.button('predict'):
        st.write('probability of Heart Disease is', output)







