import numpy as np
from art.attacks.evasion import CarliniL2Method, DeepFool, SaliencyMapMethod
from art.estimators.classification import KerasClassifier
from art.utils import load_dataset
from tensorflow.keras.models import load_model
import sys

def run_attack(classifier, x_test, y_test, attack, attack_name):
    x_test_adv = attack.generate(x=x_test)
    predictions = classifier.predict(x_test_adv)
    accuracy = np.mean(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1))
    print(f"{attack_name} - Adversarial accuracy: {accuracy:.4f}")
    return accuracy

def main():
    model = load_model('path/to/your/model.h5')
    (_, _), (x_test, y_test), min_, max_ = load_dataset('mnist')
    classifier = KerasClassifier(model=model, clip_values=(min_, max_))

    attacks = [
        (CarliniL2Method(classifier, max_iter=100), "Carlini & Wagner L2"),
        (DeepFool(classifier), "DeepFool"),
        (SaliencyMapMethod(classifier), "JSMA")
    ]

    results = []
    for attack, name in attacks:
        results.append(run_attack(classifier, x_test, y_test, attack, name))

    if min(results) < 0.5:
        print("Adversarial testing failed. Model is vulnerable to attacks.")
        sys.exit(1)
    print("Adversarial testing passed. Model is robust against tested attacks.")

if __name__ == "__main__":
    main()
