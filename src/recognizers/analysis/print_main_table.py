import argparse
import pathlib
import re

def aggregate_results(model_type, validation_type, file_path):
    data = []
    with open(file_path, 'r') as file:
        latex_lines = file.readlines()
        latex_lines = [re.sub(r'\\textbf\{(.*?)\}', r'\1', line) for line in latex_lines]
        
    for line in latex_lines:
        if "Train" in line or "Model" in line:
            continue

        if model_type in line and validation_type in line:
            parts = line.rstrip(' \\\\\n').split('&')
            model_name = parts[0]
            l_test_mean = float(parts[6].split(' $\\pm$ ', 1)[0].strip())
            l_test_max = float(parts[7].strip())
            data.append({
                "Model": model_name,
                "L. Test (Mean)": l_test_mean,
                "L. Test (Max)": l_test_max
            })

    if data:
        
        best_loss_term = "\\recognitionabbrev{}"

        match validation_type:
            case 'shortabbrev':
                key = 'L. Test (Mean)'
            case 'longabbrev':
                key = 'L. Test (Max)'
            case _:
                raise ValueError
        best_row_for_loss_term =  max(data, key=lambda x: x[key])

        if "languagemodelingabbrev" in best_row_for_loss_term["Model"]:
            best_loss_term += "+\\languagemodelingabbrev{}"
        if "nextsymbolsabbrev" in best_row_for_loss_term["Model"]:
            best_loss_term += "+\\nextsymbolsabbrev{}"

        return best_row_for_loss_term[key], best_loss_term
    else:
        return None, None

def format_score(x):
    return f'{x:.2f}'

def generate_latex_table(language_file_paths):
    model_types = ['transformerabbrev', 'rnnabbrev', 'lstmabbrev']
    validation_types = ['shortabbrev', 'longabbrev']
    
    all_results = {}
    all_best_loss_term_results = {}

    for language, file_path in language_file_paths.items():
        results = {model: {'Short': None, 'Long': None} for model in model_types}
        best_loss_term_results = {model: {'Short': None, 'Long': None} for model in model_types}

        for model in model_types:
            for val_type in validation_types:
                key = 'Short' if val_type == 'shortabbrev' else 'Long'
                results[model][key], best_loss_term_results[model][key] = aggregate_results(model, val_type, file_path)
        
        all_results[language] = results
        all_best_loss_term_results[language] = best_loss_term_results

    latex_table = "\\begin{tabular}{@{}lcccccc@{}}\n\\toprule\n"
    latex_table += "& \\multicolumn{3}{c}{Inductive Bias} & \\multicolumn{3}{c}{Expressivity} \\\\\n"
    latex_table += "\\cmidrule(lr){2-4} \\cmidrule(lr){5-7}\n"
    latex_table += "Language & \\transformerabbrev{} & \\rnnabbrev{} & \\lstmabbrev{} & \\transformerabbrev{} & \\rnnabbrev{} & \\lstmabbrev{} \\\\\n"
    latex_table += "\\midrule\n"

    for language, results in all_results.items():
        
        short_vals = [results[model]['Short'] if results[model]['Short'] is not None else 'N/A' for model in model_types]
        long_vals = [results[model]['Long'] if results[model]['Long'] is not None else 'N/A' for model in model_types]
        latex_table += f"{language} & {' & '.join(map(format_score, short_vals))} & {' & '.join(map(format_score, long_vals))} \\\\\n"

    latex_table += "\\bottomrule\n\\end{tabular}\n"

    second_latex_table = "\\begin{tabular}{@{}lcccccc@{}}\n\\toprule\n"
    second_latex_table += "& \\multicolumn{3}{c}{Inductive Bias} & \\multicolumn{3}{c}{Expressivity} \\\\\n"
    second_latex_table += "\\cmidrule(lr){2-4} \\cmidrule(lr){5-7}\n"
    second_latex_table += "Language & \\transformerabbrev{} & \\rnnabbrev{} & \\lstmabbrev{} & \\transformerabbrev{} & \\rnnabbrev{} & \\lstmabbrev{} \\\\\n"
    second_latex_table += "\\midrule\n"

    for language, results in all_best_loss_term_results.items():
        short_vals = [results[model]['Short'] if results[model]['Short'] is not None else 'N/A' for model in model_types]
        long_vals = [results[model]['Long'] if results[model]['Long'] is not None else 'N/A' for model in model_types]
        second_latex_table += f"{language} & {' & '.join(short_vals)} & {' & '.join(long_vals)} \\\\\n"

    second_latex_table += "\\bottomrule\n\\end{tabular}\n"

    return latex_table, second_latex_table

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--language', nargs=2, action='append', default=[])
    parser.add_argument('--main-output', type=pathlib.Path, required=True)
    parser.add_argument('--loss-output', type=pathlib.Path, required=True)
    args = parser.parse_args()

    language_file_paths = { language : path for language, path in args.language }

    first_latex_table, second_latex_table = generate_latex_table(language_file_paths)

    print(f'writing {args.main_output}')
    with args.main_output.open('w') as file:
        file.write(first_latex_table)

    print(f'writing {args.loss_output}')
    with args.loss_output.open('w') as file:
        file.write(second_latex_table)

if __name__ == '__main__':
    main()
