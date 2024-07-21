import torch
import d2l
@d2l.add_to_class(d2l.Trainer)
def prepare_batch(self, batch):
    return batch

@d2l.add_to_class(d2l.Trainer)
def fit_epoch(self):
    self.model.train() # activate training mode
    for batch in self.train_dataloader:
        # calculate loss for batch
        loss = self.model.training_step(self.prepare_batch(batch))
        # set gradients to zero
        self.optim.zero_grad()
        # w/ disable gradient calculation (to reduce memory use)
        with torch.no_grad():
            # compute gradients of loss wrt params
            loss.backward()
            if self.gradient_clip_val > 0:
                self.clip_gradients(self.gradient_clip_val, self.model)
            # update model params w/ gradients
            self.optim.step()
        # increment by 1 - tracking purposes
        self.train_batch_idx += 1
    # check if validation, if not return early
    if self.val_dataloader is None:
        return
    self.model.eval()
    for batch in self.val_dataloader:
        with torch.no_grad():
            self.model.validation_step(self.prepare_batch(batch))
        self.val_batch_idx += 1