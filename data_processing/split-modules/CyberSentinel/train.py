import tensorflow as tf

def compile_and_train(model, train_data, train_labels, config):
    # Learning rate schedule
    if config.get('learning_rate_schedule'):
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=config['learning_rate_schedule']['initial_learning_rate'],
            decay_steps=config['learning_rate_schedule']['decay_steps'],
            decay_rate=config['learning_rate_schedule']['decay_rate']
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule) # type:ignore
    else:
        optimizer = config.get('optimizer', 'adam')
    
    # Custom metrics
    custom_metrics = config.get('custom_metrics', ['accuracy'])
    
    # Compile model
    model.compile(loss=config.get('loss', 'binary_crossentropy'), optimizer=optimizer, metrics=custom_metrics)
    
    # Callbacks
    callbacks = []
    
    # Early stopping
    if config.get('early_stopping'):
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor=config['early_stopping']['monitor'],
            patience=config['early_stopping']['patience']
        )
        callbacks.append(early_stopping)
    
    # Train model
    history = model.fit(
        train_data,
        train_labels,
        epochs=config.get('epochs', 10),
        validation_split=config.get('validation_split', 0.2),
        callbacks=callbacks
    )
    
    return history
