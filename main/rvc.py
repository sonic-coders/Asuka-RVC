from rvc import Config, run_inference_script


def infer(input_path,
          model_path,
          output_path, 
          pitch=0,
          f0_method="rmvpe",
          embedder_model="contentvec_base", 
          index_path=None, 
          index_rate=0.5,
          log_level="info"
         ):
    """Convert a single audio file using the function-style API.

    Note: this is the SAME as examples/infer.py functionally — the only
    difference is the API style. New code should prefer the class form
    (RVClass) so the model can be reused across multiple conversions.
    """
    config = Config(
        embedder_model=embedder_model,
        f0_method=f0_method,
        log_level=log_level,
    )

    # One-shot function call. Internally creates an RVClass, runs the
    # conversion, and cleans up — but the model is loaded only once.
    run_inference_script(
        config=config,
        input_path=input_path,
        output_path=output_path,
        pth_path=model_path,
        pitch=pitch,
        f0_method=f0_method,
        index_path=index_path,
        index_rate=index_rate,
    )

    print(f"Done! Output saved to: {output_path}")
    return True
