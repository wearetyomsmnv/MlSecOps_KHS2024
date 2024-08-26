import numpy as np
from art.attacks.evasion import ProjectedGradientDescent
from art.estimators.classification import KerasClassifier
from art.utils import load_dataset
from tensorflow.keras.models import load_model
import sys

def main():
    model = load_model('path/to/your/model.h5')
    (_, _), (x_test, y_test), min_, max_ = load_dataset('mnist')
    classifier = KerasClassifier(model=model, clip_values=(min_, max_))
    
    attack = ProjectedGradientDescent(estimator=classifier, eps=0.2, eps_step=0.1, max_iter=10)
    x_test_adv = attack.generate(x=x_test)
    predictions = classifier.predict(x_test_adv)
    accuracy = np.mean(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1))
    
    print(f"Adversarial accuracy: {accuracy}")
    if accuracy < 0.5:
        print("Adversarial testing failed. Model is vulnerable to attacks.")
        sys.exit(1)
    print("Adversarial testing passed. Model is robust.")

if __name__ == "__main__":
    main()
