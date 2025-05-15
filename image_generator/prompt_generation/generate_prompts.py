import os
import openai
import time

def process_file(input_path, output_path, client, model):
    # Read the content of the input file
    with open(input_path, "r", encoding="utf-8") as file:
        prompt = file.read()

    # Send the request to OpenAI API
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=500  # Limit the response to a maximum of 100 tokens per element list (5*100)
    )

    # Extract the response text
    response_text = response.choices[0].message.content
    response_text = response_text.replace("```python\n", "").replace("\n```", "").replace("prompts = ", "")

    # Save the response to a new file
    with open(output_path, "w", encoding="utf-8") as file:
        file.write(response_text)

    print(f"Response saved to: '{output_path}'")
    time.sleep(1)
            

def generate_prompts(input_directory):
    # Load OpenAI API key from environment variables
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key not found. Set the OPENAI_API_KEY environment variable. COPY THIS:\n\nexport $OPENAI_API_KEY=")

    # Initialize OpenAI client
    client = openai.Client(api_key=api_key)

    # Define the model to use
    model = "gpt-4o"

    # Process each file in the directory
    for filename in os.listdir(input_directory):
        if filename.endswith(".txt") and not filename.endswith("_response.txt"):
            # print(f"Processing file '{filename}'..")

            input_path = os.path.join(input_directory, filename)
            output_filename = filename.replace(".txt", "_response.txt")
            output_path = os.path.join(input_directory, output_filename)

            # skip if _response file already exists!
            if not os.path.isfile(output_path):
                process_file(input_path, output_path, client, model)
            else:
                print("RESPONSE ALREADY EXISTS! SKIPPING ..")
            
            # print(50*"*")
            
            
def main():
    # Load OpenAI API key from environment variables
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key not found. Set the OPENAI_API_KEY environment variable. COPY THIS:\n\nexport $OPENAI_API_KEY=")

    # Initialize OpenAI client
    client = openai.Client(api_key=api_key)

    # Set the directory containing the input text files
    input_directory = "../data/output_folder"

    # Define the model to use
    model = "gpt-4o"

    # Process each file in the directory
    for filename in os.listdir(input_directory):
        if filename.endswith(".txt") and not filename.endswith("_response.txt"):
            print(f"Processing file '{filename}'..")

            input_path = os.path.join(input_directory, filename)
            output_filename = filename.replace(".txt", "_response.txt")
            output_path = os.path.join(input_directory, output_filename)

            # skip if _response file already exists!
            if not os.path.isfile(output_path):
                process_file(input_path, output_path, client, model)
            else:
                print("RESPONSE ALREADY EXISTS! SKIPPING ..")
            
            print(50*"*")



if __name__ == "__main__":
    main()
    print()
    print()
    print("Process finished!")