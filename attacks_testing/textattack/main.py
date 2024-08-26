import sys
import textattack
from textattack.models.wrappers import HuggingFaceModelWrapper
from textattack.datasets import HuggingFaceDataset
from textattack.attack_recipes import (
    TextFoolerJin2019, 
    BERTAttackLi2020, 
    PWWSRen2019
)
from textattack.attackers import Attacker
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import junit_xml

def run_attack(model_wrapper, dataset, attack_recipe, attack_name):
    attack = attack_recipe.build(model_wrapper)
    attacker = Attacker(attack, dataset)
    results = attacker.attack_dataset()
    
    success_rate = results.success_rate()
    print(f"{attack_name} - Attack Success Rate: {success_rate:.4f}")
    return success_rate, attack_name

def main():
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_wrapper = HuggingFaceModelWrapper(model, tokenizer)

    dataset = HuggingFaceDataset("glue", "sst2", split="validation")

    attacks = [
        (TextFoolerJin2019, "TextFooler"),
        (BERTAttackLi2020, "BERT-Attack"),
        (PWWSRen2019, "PWWS")
    ]

    test_cases = []
    overall_success = True
    for attack_recipe, name in attacks:
        success_rate, attack_name = run_attack(model_wrapper, dataset, attack_recipe, name)
        test_case = junit_xml.TestCase(attack_name, classname="AdversarialTest")
        if success_rate > 0.5:
            test_case.add_failure_info(f"Attack success rate too high: {success_rate:.4f}")
            overall_success = False
        test_cases.append(test_case)

    test_suite = junit_xml.TestSuite("AdversarialTestSuite", test_cases)
    with open('test-results.xml', 'w') as f:
        junit_xml.TestSuite.to_file(f, [test_suite], prettyprint=True)

    if not overall_success:
        print("Adversarial testing failed. Model is vulnerable to attacks.")
        sys.exit(1)
    print("Adversarial testing passed. Model is relatively robust against tested attacks.")

if __name__ == "__main__":
    main()
