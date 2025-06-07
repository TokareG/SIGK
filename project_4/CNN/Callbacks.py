from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

def get_early_stopping():
    early_stop_callback = EarlyStopping(
       monitor='val_loss',
       patience=15,
       verbose=False,
       mode='min'
    )
    return early_stop_callback

def get_checkpoint_callback():
    MODEL_CKPT_PATH = 'CNN_model/'
    MODEL_CKPT = 'model-{epoch:02d}-{val_loss:.2f}'

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=MODEL_CKPT_PATH,
        filename=MODEL_CKPT,
        save_top_k=20,
        mode='min'
    )
    return checkpoint_callback