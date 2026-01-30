"""
AzSentiment - Streamlit Application
Interactive sentiment analysis for Azerbaijani text.
"""
import streamlit as st
from transformers import pipeline

# Page config
st.set_page_config(
    page_title="AzSentiment Analyzer",
    page_icon="AZ",
    layout="centered"
)


@st.cache_resource
def load_model():
    """Load the fine-tuned sentiment model."""
    return pipeline(
        "text-classification",
        model="./models/az-sentiment-best"
    )


def main():
    st.title("Azerbaijani Sentiment Analyzer")
    st.markdown("""
    Analyze the sentiment of Azerbaijani text using a fine-tuned aLLMA-BASE model.
    """)
    
    # Input
    text = st.text_area(
        "Enter Azerbaijani text:",
        placeholder="Meseleni: Bu mehsul cox yaxsidir!",
        height=100
    )
    
    # Analyze button
    if st.button("Analyze Sentiment", type="primary"):
        if text.strip():
            with st.spinner("Analyzing..."):
                try:
                    classifier = load_model()
                    result = classifier(text)[0]
                    
                    label = result["label"]
                    score = result["score"]
                    
                    # Display result
                    if "POSITIVE" in label.upper() or "1" in label:
                        sentiment = "Positive"
                        color = "green"
                    else:
                        sentiment = "Negative"
                        color = "red"
                    
                    st.markdown(f"""
                    ### Result
                    <div style="padding: 20px; border-radius: 10px; background-color: {'#d4edda' if color == 'green' else '#f8d7da'};">
                        <h2 style="margin: 0;">{sentiment}</h2>
                        <p style="margin: 5px 0 0 0;">Confidence: <strong>{score:.1%}</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"Error loading model: {e}")
                    st.info("Make sure the model is trained and saved in `models/az-sentiment-best/`")
        else:
            st.warning("Please enter some text to analyze.")
    
    # Example texts
    with st.expander("Example Texts"):
        st.markdown("""
        **Positive examples:**
        - Bu restoran cox yaxsidir! Yemekler lezzetli idi.
        - Mehsul ela keyfiyyetdedir, cox raziyam.
        - Xidmet mukemmel idi, tovsiye edirem.
        
        **Negative examples:**
        - Xidmet cox pis idi, hec vaxt geri qayitmayacagam.
        - Mehsul siniq geldi, keyfiyyet cox asagidir.
        - Pul itkisi, almayin.
        """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "Model: `allmalab/bert-base-aze` (aLLMA-BASE) | "
        "[Paper](https://aclanthology.org/2024.sigturk-1.2)"
    )


if __name__ == "__main__":
    main()
