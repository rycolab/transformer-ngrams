import numpy as np


def H(x: np.ndarray) -> np.ndarray:
    """The Heaviside (threshold) function."""
    return (x > 0.0).astype(np.int_)


def saturated_sigmoid(x: np.ndarray) -> np.ndarray:
    """The saturated sigmoid function."""
    return np.clip(x, 0, 1)


def ReLU(x: np.ndarray) -> np.ndarray:
    """The ReLU function."""
    return np.maximum(x, 0)


class ProjectionFunctions:
    @staticmethod
    def soft(s: np.ndarray) -> np.ndarray:
        """Applies the projection function of soft attention mechanism
            to the input scores.

        Args:
            s (np.ndarray): The input scores.

        Returns:
            np.ndarray: The output scores.
        """
        return np.exp(s) / np.sum(np.exp(s))

    @staticmethod
    def averaging_hard(s: np.ndarray) -> np.ndarray:
        """Applies the projection function of averaging hard attention mechanism
            to the input scores.

        Args:
            s (np.ndarray): The input scores.

        Returns:
            np.ndarray: The output scores.
        """
        a = s == np.max(s)
        return np.asarray(a, dtype=np.float32) / np.sum(a)

    @staticmethod
    def unique_hard(s: np.ndarray) -> np.ndarray:
        """Applies the projection function of unique hard attention mechanism
            to the input scores.

        Args:
            s (np.ndarray): The input scores.

        Returns:
            np.ndarray: The output scores.
        """
        a = np.zeros_like(s)
        a[np.argmax(s)] = 1
        return a
