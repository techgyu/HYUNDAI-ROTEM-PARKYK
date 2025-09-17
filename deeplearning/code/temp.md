Traceback (most recent call last):
  File "c:\Users\htw02\HDRT_AI\PART2\sigmoid\tf26.py", line 117, in <module>
    history = model.fit(x_train,y_train,batch_size=8,epochs=50,verbose=0,validation_split=0.2)
  File "c:\Users\htw02\anaconda3\envs\taewoo1\lib\site-packages\keras\utils\traceback_utils.py", line 70, in error_handler
    raise e.with_traceback(filtered_tb) from None
  File "C:\Users\htw02\AppData\Local\Temp\__autograph_generated_file4n60761k.py", line 15, in tf__train_function
    retval_ = ag__.converted_call(ag__.ld(step_function), (ag__.ld(self), ag__.ld(iterator)), None, fscope) 
ValueError: in user code:

    File "c:\Users\htw02\anaconda3\envs\taewoo1\lib\site-packages\keras\engine\training.py", line 1160, in train_function  *
        return step_function(self, iterator)
    File "c:\Users\htw02\anaconda3\envs\taewoo1\lib\site-packages\keras\engine\training.py", line 1146, in step_function  **
        outputs = model.distribute_strategy.run(run_step, args=(data,))
    File "c:\Users\htw02\anaconda3\envs\taewoo1\lib\site-packages\keras\engine\training.py", line 1135, in run_step  **
        outputs = model.train_step(data)
    File "c:\Users\htw02\anaconda3\envs\taewoo1\lib\site-packages\keras\engine\training.py", line 994, in train_step
        loss = self.compute_loss(x, y, y_pred, sample_weight)
    File "c:\Users\htw02\anaconda3\envs\taewoo1\lib\site-packages\keras\engine\training.py", line 1052, in compute_loss
        return self.compiled_loss(
    File "c:\Users\htw02\anaconda3\envs\taewoo1\lib\site-packages\keras\engine\compile_utils.py", line 265, in __call__
        loss_value = loss_obj(y_t, y_p, sample_weight=sw)
    File "c:\Users\htw02\anaconda3\envs\taewoo1\lib\site-packages\keras\losses.py", line 152, in __call__   
        losses = call_fn(y_true, y_pred)
    File "c:\Users\htw02\anaconda3\envs\taewoo1\lib\site-packages\keras\losses.py", line 272, in call  **   
        return ag_fn(y_true, y_pred, **self._fn_kwargs)
    File "c:\Users\htw02\anaconda3\envs\taewoo1\lib\site-packages\keras\losses.py", line 1990, in categorical_crossentropy
        return backend.categorical_crossentropy(
    File "c:\Users\htw02\anaconda3\envs\taewoo1\lib\site-packages\keras\backend.py", line 5529, in categorical_crossentropy
        target.shape.assert_is_compatible_with(output.shape)

    ValueError: Shapes (None, 3) and (None, 10) are incompatible


(taewoo1) c:\Users\htw02\HDRT_AI\PART2\sigmoid>