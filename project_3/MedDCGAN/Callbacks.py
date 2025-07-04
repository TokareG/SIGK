from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

def get_early_stopping():
    early_stop_callback = EarlyStopping(
       monitor='val_loss',
       patience=10,
       verbose=False,
       mode='min'
    )
    return early_stop_callback

def get_checkpoint_callback(class_id=None):
    if class_id is None:
        MODEL_CKPT_PATH = 'DCGAN_model/'
    else:
        MODEL_CKPT_PATH = f'DCGAN_model/class_{class_id}'
    MODEL_CKPT = 'model-{epoch:02d}-{val_loss:.2f}'

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=MODEL_CKPT_PATH,
        filename=MODEL_CKPT,
        save_top_k=10,
        mode='min'
    )
    return checkpoint_callback