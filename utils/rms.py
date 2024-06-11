import jax
import jax.numpy as jnp
import flax

from utils.train_state import nonpytree_field

class RunningMeanStd(flax.struct.PyTreeNode):
    eps: jnp.array = jnp.array(jnp.finfo(jnp.float32).eps.item(),dtype=jnp.float32)
    mean: jnp.array = 1.0
    var: jnp.array = 1.0
    clip_max: jnp.array = jnp.array(10.0,dtype=jnp.float32)
    count: jnp.array = jnp.array(0,dtype=jnp.int32)
    only_var: bool = nonpytree_field(default=False)

    def norm(self, data_array):
        if not self.only_var:
            data_array = (data_array - self.mean) / jnp.sqrt(self.var + self.eps)
            data_array = jnp.clip(data_array, -self.clip_max, self.clip_max)
        else:
            data_array = data_array / jnp.sqrt(self.var + self.eps)
            data_array = jnp.clip(data_array, -self.clip_max, self.clip_max)
        return data_array
    
    def denorm(self, data_array):
        if not self.only_var:
            return data_array * jnp.sqrt(self.var + self.eps) + self.mean
        else:
            return data_array * jnp.sqrt(self.var + self.eps)

    def update(self, data_array: jnp.ndarray) -> None:
        """Add a batch of item into RMS with the same shape, modify mean/var/count."""
        batch_mean, batch_var = jnp.mean(data_array, axis=0), jnp.var(data_array, axis=0)
        batch_count = len(data_array)

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        new_var = m_2 / total_count

        return self.replace(
            mean=new_mean,
            var=new_var,
            count=total_count
        )