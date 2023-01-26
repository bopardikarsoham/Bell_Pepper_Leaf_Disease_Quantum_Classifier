import streamlit as st
from PIL import Image
from Predict_Function import predict

html_temp = """
    <div>
    <h2 style="color:MEDIUMSEAGREEN;text-align:left;"> Bell Pepper Leaf Disease Detection 🍀</h2>
    </div>
    """
st.markdown(html_temp, unsafe_allow_html=True)

col1,col2  = st.columns([2,2])
    
with col1: 
    with st.expander(" ℹ️ Information", expanded=True):
        st.write("""
        Farming dominates as an occupation in the agriculture domain in more than 125 countries. However, even these crops are, subjected to infections and diseases. Plant diseases are a major threat to food security at the global scale. Plant diseases are a significant threat to human life as they may lead to 
        droughts and famines, due to rapid infection and lack of the necessary infrastructure. It's troublesome to observe the plant diseases manually. It needs tremendous quantity of labor, expertise within the plant diseases. Here I present to you a hybrid quantum-classical Deep Learning Model that solves the problem for a Bell Pepper Leaf. 
        """)
        '''
        ## How does it work ❓ 
        Upload an image of a single bell pepper leaf and the model will predict whether it is healthy or diseased.
        '''

    with col2:
        
        st.set_option('deprecation.showfileUploaderEncoding', False)
        st.subheader("Check the freshness of your leaves 👨‍🌾")

        file_up = st.file_uploader("Upload an image", type="jpg")

        if file_up is not None:
            image = Image.open(file_up)
            st.image(image, caption='Uploaded Image.', use_column_width=True)
            st.write("")
            st.write("Please wait...")
            labels = predict(file_up)

            # print out the top 5 prediction labels with scores
            for i in labels:
                st.write("Prediction (index, name)", i[0], ",   Score: ", i[1])