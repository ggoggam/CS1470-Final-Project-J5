import tensorflow as tf
import time

class RunManager:
    def __init__(self, model):
        self.model = model
        self.lastTime = time.time()
        self.time = time.time()
        self.lastOutput = -1
        self.lastState = [[tf.zeros([1,128]), tf.zeros([1, 128])] for _ in range(2)]
    
    def getInputFeats(self, button):
        featsArr = []

        buttonTensor = tf.convert_to_tensor([button], dtype=tf.float32)
        buttonScaled = tf.math.subtract(tf.math.multiply(2., tf.math.divide(buttonTensor, 7)), 1)
        featsArr.append(buttonScaled)
        
        lastOutputOh = tf.one_hot(self.lastOutput + 1, 89, dtype=tf.float32)
        featsArr.append(lastOutputOh)

        self.time = time.time()
        deltaTime = self.time - self.lastTime
        deltaTimeBin = tf.math.round(deltaTime * 31.25)

        deltaTimeTrunc = min(deltaTimeBin, 32)
        # deltaTimeInt = tf.cast(tf.math.add(deltaTimeTrunc, 1e-4), 'tf.int32')
        deltatimeOh = tf.one_hot(deltaTimeTrunc, 33, dtype=tf.float32)
        featsArr.append(deltatimeOh)
        self.lastTime = self.time

        return tf.concat(featsArr, axis=-1)

    def next(self, button):
        input_feats = self.getInputFeats(button)
        input_feats = tf.reshape(input_feats, [1, 1, -1])
        print(tf.shape(input_feats))

        logits, state = self.model.evaluate(input_feats, self.lastState)
        note = tf.math.argmax(logits[0][0])

        self.lastOutput = note
        self.lastState = state

        return note
