import tensorflow as tf
from tensorflow import keras
from easyGame import EasyGame
import numpy as np

print('Tensorflow loaded.')


def getModel():
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(
        16, (3, 3), padding='same', input_shape=[84, 84, 1]))
    model.add(keras.layers.LeakyReLU(alpha=0.3))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Dropout(0.25))

    model.add(keras.layers.Conv2D(32, (3, 3), padding='same'))
    model.add(keras.layers.LeakyReLU(alpha=0.3))
    model.add(keras.layers.Dropout(0.25))

    model.add(keras.layers.Conv2D(32, (3, 3), padding='same'))
    model.add(keras.layers.LeakyReLU(alpha=0.3))
    model.add(keras.layers.Dropout(0.25))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(256))
    model.add(keras.layers.LeakyReLU(alpha=0.3))

    model.add(keras.layers.Dropout(0.5))

    model.add(keras.layers.Dense(128))
    model.add(keras.layers.LeakyReLU(alpha=0.3))

    model.add(keras.layers.Dense(64))
    model.add(keras.layers.LeakyReLU(alpha=0.3))

    model.add(keras.layers.Dense(32))
    model.add(keras.layers.LeakyReLU(alpha=0.3))

    model.add(keras.layers.Dense(3))

    model.compile(loss='mean_squared_error',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


model = getModel()
Q = model.predict
model.summary()

dis = 0.9

stateList = []
rewardList = []

for epi in range(1000):
    # Do a game

    myGame = EasyGame()
    state = np.zeros([84, 84, 1])

    while True:
        # Do a action with current state
        action = Q(np.array([state]))[0] + np.random.randn(3) / (epi/10+1)
        assert not np.isnan(action).any()

        # Do some steps with given action.
        for _ in range(3):
            # The state is updated and the reward is given by the environment.
            newState, gameEnd, time = myGame.step(action)
            if gameEnd:
                break

        if gameEnd:
            reward = -100
            rewardVector = np.copy(action)
            rewardVector[np.argmax(action)] = reward

        else:
            reward = 1
            rewardVector = np.copy(action)
            rewardVector[np.argmax(action)] = reward + dis * \
                np.max(Q(np.array([newState])))

        print(time, myGame.player['x'], rewardVector)

        stateList.append(state)
        rewardList.append(rewardVector)

        # Update state
        state = newState

        if gameEnd:
            break

    # Queue
    if len(stateList) > 1000:
        stateList = stateList[-999:]
        rewardList = rewardList[-999:]

    # Train
    print('Epi :', epi)
    print('Dataset :', len(stateList))
    train_x = np.array(stateList)
    train_y = np.array(rewardList)
    model.fit(train_x, train_y)

    # Save model
    if epi % 5 == 0:
        model_json = model.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)
        print("Model structure saved.")
        model.save_weights("model.h5")
        print("Model weight saved.")
