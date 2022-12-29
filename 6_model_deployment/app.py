import streamlit as st
import pickle
import numpy as np

# Stores loaded model in cache so that we don't need to reload model repeatedly for each input
@st.cache(allow_output_mutation=True) 
def load_model():
	model = pickle.load(open('random_forest_model.sav', 'rb'))
	country_dict = pickle.load(open('country_dict.pickle', 'rb'))
	scaler = pickle.load(open('standardScaler.pickle', 'rb'))
	return model, scaler, country_dict

def featurize(time, country, scaler, country_dict):
	arr = np.array([country_dict[country], time]).reshape(1,-1)
	vector = scaler.transform(arr)
	return vector

def main():
	model, scaler, country_dict = load_model()
	st.title("\'365 data science\' : free-to-paid user conversion predictor")
	list_of_countries = list(country_dict.keys())
	st.write("\'365 data science\' is a ed-tech company that creates data science courses comprising of video lectures and \
		exercises in the form of quizzes and exams. Some of the courses offered are free and majority of the other courses \
		need the user to buy paid subscription. Students mostly register on this platform as 'free-tier user' as the registration is free of cost. \
		They enroll for free courses and then if they like the content of the platform, they proceed to buy paid-subscription \
		which offers lot of perks as compared to free tier. Paid student get access to large library of courses along with certificates, \
		quizzes and exams.")
	st.write("This application predicts how likely the student is to buy the paid subscription based on the number of minutes \
		he spent engaging with the free course content and the country he comes from. In the exploratory data analysis done, it was found that \
		total time spent by user and nationality of user are two major and most significant factor for determining how likely the user is \
		to buy the course. Typical range for total time watched for students is mostly 0.1 to 100 minutes")
	
	with st.form("my_form"):
		total_time = st.number_input('Time spent on platform watching tutorials')
		student_country = st.selectbox('country', list_of_countries)
		st.write('Total time spent : ', total_time)
		st.write('Student country :', student_country)

		# Every form must have a submit button.
		submitted = st.form_submit_button("Submit")
	
	if submitted: 
		vector = featurize(total_time, student_country, scaler, country_dict)
		prediction = model.predict(vector)[0]
		predicted_proba = model.predict_proba(vector)
		if prediction == 0 : 
			st.write('Student is ', str(round(predicted_proba[0][0]*100)), '%  likely to NOT buy the paid subscription')
		else : 
			st.write('Student is ', str(round(predicted_proba[0][1]*100)), '%  likely to buy the paid subscription')

if __name__ == '__main__' : 
	main()