# Project Title

This project is designed to evaluate the performance of language models using a set of predefined test cases. The evaluation process involves generating responses from the models and scoring them based on specific criteria.

## Prerequisites

- Python 3.x
- Required Python packages (listed in `requirements.txt`)

## Installation

1. Clone the repository:
    ```sh
    git clone <repository-url>
    cd <repository-directory>
    ```

2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. Prepare a YAML configuration file with the necessary settings. An example configuration file might look like this:

    ```yaml
    evaluation_model:
      args:
        base_url: "http://example.com"
        token: "your_token"
      body_args:
        model_name: "model_name"
        system_prompt: "system_prompt"

    evaluation_criteria:
      evals: ["criteria1", "criteria2"]

    test_examples:
      group1:
        - text: "example question 1"
          reply: "expected reply 1"
      group2:
        - text: "example question 2"
          reply: "expected reply 2"
    ```
    or
    2. Here is another example YAML configuration file that does not include the `reply` field:

    ```yaml
    response_model:
      args:
        base_url: "http://example.com"
        token: "your_token"
      body_args:
        model_name: "model_name"
        system_prompt: "system_prompt"

    evaluation_model:
      args:
        base_url: "http://example.com"
        token: "your_token"
      body:
        model_name: "model_name"
        system_prompt: "system_prompt"

    evaluation_criteria:
      evals: ["criteria1", "criteria2"]

    test_examples:
      group1:
        - text: "example question 1"
      group2:
        - text: "example question 2"
    ```

2. Run the script with the YAML configuration file:
    ```sh
    python run_task.py --yaml path/to/your/config.yaml
    ```


## Output

The script will generate a CSV file (`output_file.csv`) containing the results of the evaluation, including the model name, prompt, group, text, reply, score, and reason.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.