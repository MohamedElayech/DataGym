def generate_ai_interpretation(model_name, metrics_results, classification_report, hf_token=None):
    """Generate an AI interpretation of the model results using a Hugging Face model (no pipeline)"""
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM 
        import torch
        # Prepare the prompt with model results
        metrics_text = ", ".join([f"{k}: {v:.4f}" for k, v in metrics_results.items()])
        prompt = f"""Analyze the following machine learning model results:\n\nModel: {model_name}\nMetrics: {metrics_text}\n\nProvide a detailed analysis including:\n1. Performance evaluation\n2. Strengths and weaknesses\n3. Recommendations for improvement\n\nMake a summarization\n\nAnalysis:"""
        
        # Use a Hugging Face model for text generation (no pipeline)
        model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # You can change this to any supported text2text model
        if hf_token:
            tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=hf_token)
            model = AutoModelForCausalLM .from_pretrained(model_id, use_auth_token=hf_token)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(model_id)

        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        # Generate output
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_length=200, do_sample=True, top_p=0.95, temperature=0.7)
        ai_interpretation = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print(ai_interpretation)
        # Format the final interpretation
        interpretation = f"### AI-Generated Performance Analysis\n\n{ai_interpretation}\n\n"
        
        # Add some standard analysis based on the metrics
        interpretation += "### Quick Metrics Summary\n\n"
        if "Accuracy" in metrics_results:
            accuracy = metrics_results["Accuracy"]
            if accuracy > 0.9:
                interpretation += f"✅ **Excellent Performance**: Accuracy of {accuracy:.2%}\n"
            elif accuracy > 0.7:
                interpretation += f"✓ **Good Performance**: Accuracy of {accuracy:.2%}\n"
            else:
                interpretation += f"⚠ **Needs Improvement**: Accuracy of {accuracy:.2%}\n"
        if "Precision" in metrics_results and "Recall" in metrics_results:
            precision = metrics_results["Precision"]
            recall = metrics_results["Recall"]
            interpretation += f"- **Precision**: {precision:.2%} | **Recall**: {recall:.2%}\n"
        interpretation += "\n### Recommendations\n"
        interpretation += "- Review the confusion matrix for error patterns\n"
        interpretation += "- Consider feature engineering if needed\n"
        interpretation += "- Try different algorithms for comparison\n"
        return interpretation
    except Exception as e:
        # Fallback to rule-based interpretation if AI generation fails
        print(f"AI generation error: {str(e)}. Using fallback interpretation.")
        metrics_text = ", ".join([f"{k}: {v:.4f}" for k, v in metrics_results.items()])
        interpretation = f"### Performance Analysis\n\n"
        interpretation += f"The {model_name} model achieved the following metrics: {metrics_text}.\n\n"
        if "Accuracy" in metrics_results:
            accuracy = metrics_results["Accuracy"]
            if accuracy > 0.9:
                interpretation += f"✅ **Excellent Performance**: The model shows excellent accuracy ({accuracy:.2%}), correctly classifying most samples.\n\n"
            elif accuracy > 0.7:
                interpretation += f"✓ **Good Performance**: The model shows good accuracy ({accuracy:.2%}), but there is room for improvement.\n\n"
            else:
                interpretation += f"⚠ **Needs Improvement**: The model's accuracy ({accuracy:.2%}) suggests it may be struggling with this dataset.\n\n"
        if "Precision" in metrics_results and "Recall" in metrics_results:
            precision = metrics_results["Precision"]
            recall = metrics_results["Recall"]
            interpretation += f"**Precision vs Recall Balance**:\n"
            interpretation += f"- Precision: {precision:.2%}\n"
            interpretation += f"- Recall: {recall:.2%}\n\n"
            if abs(precision - recall) < 0.05:
                interpretation += "The model shows a balanced approach between precision and recall.\n\n"
            elif precision > recall:
                interpretation += "The model is more conservative, prioritizing precision (fewer false positives) over recall.\n\n"
            else:
                interpretation += "The model prioritizes recall (catching all positive cases) over precision.\n\n"
        interpretation += "\n### Recommendations\n"
        interpretation += "- Review the confusion matrix for detailed error patterns\n"
        interpretation += "- Consider feature engineering if performance is suboptimal\n"
        interpretation += "- Try different hyperparameters or algorithms for comparison\n"
        return interpretation