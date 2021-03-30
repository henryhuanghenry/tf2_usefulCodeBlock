# F1 score of multiple class classification.

单标签多分类问题的F1分数计算代码

---

## MacroF1

```python
import tensorflow_addons as tfa
import tensorflow as tf
# y_true is onehot encoded, y_pred is the output of softmax(The hidden unit of the output layer should be equal to the number of classes.)
def macro_f1_tfaddon(y_true, y_pred, num_class):
    metric = tfa.metrics.F1Score(num_classes=num_class, average='macro')
    y_pred = tf.one_hot(tf.argmax(y_pred, 1), depth=num_class)
    metric.update_state(y_true, y_pred)
    result = metric.result()
    return result.numpy()
```

## MicroF1

```python
import tensorflow_addons as tfa
import tensorflow as tf
# y_true is onehot encoded, y_pred is the output of softmax(The hidden unit of the output layer should be equal to the number of classes.)
def micro_f1_tfaddon(y_true, y_pred, num_class):
    metric = tfa.metrics.F1Score(num_classes=num_class, average='micro')
    y_pred = tf.one_hot(tf.argmax(y_pred, 1), depth=num_class)
    metric.update_state(y_true, y_pred)
    result = metric.result()
    return result.numpy()
```
