import pickle

# # Load trained model once
# with open("gesture_model.pkl", "rb") as file:
#     model = pickle.load(file)


# def predict(row):

#     prediction = model.predict([row])[0]

#     confidence = max(
#         model.predict_proba([row])[0]
#     )

   
#     return prediction,float( round(confidence * 100, 2)) 

# # TEST
# # dummy_row = [0] * 63

# # result = predict(dummy_row)

# # print(result)


# if __name__ == "__main__":

#     dummy_row = [0] * 63

#     print(predict(dummy_row))
import pickle

with open("gesture_model.pkl", "rb") as file:
    model = pickle.load(file)

def predict(row):

    prediction = model.predict([row])[0]

    probabilities = model.predict_proba([row])[0]

    confidence = max(probabilities)

    return (
        prediction,
        round(confidence * 100, 2),
        probabilities,
        model.classes_
    )