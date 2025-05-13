def generate_response(query, relevant_documents):
    context = "\n".join([doc.page_content for doc in relevant_documents])  # Combine the relevant documents into context
    prompt = f"You are an experienced farmer, Given the following information \n{context}\nAnswer the following question as an experienced farmer:\n{query}, also dont mention, based on the text, just answer me like we are in a conversation"

    # Construct prompt based on the row data
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text