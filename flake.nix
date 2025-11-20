{
  description = "SpikeSynth Dev Shell";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs";

  outputs = { self, nixpkgs, ... }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };

      # This needs to be generated:  nix run github:nix-community/pip2nix -- generate -r requirements.txt --output ./python-packages.nix
      pythonOverlay = pkgs.callPackage ./python-packages.nix {};

      # Correctly override the python interpreter to create a new package set
      myPython3 = pkgs.python312.override {
        packageOverrides = pythonOverlay;
      };
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        buildInputs = [
          # Include Python with your packages
          (myPython3.withPackages (ps: with ps; [
              pandas
              requests
              numpy
              torch
              matplotlib
              jupyterlab
              polars
              tqdm
              ipywidgets
              seaborn
              lightning
              torchmetrics
	      wandb
              scipy
	      plotly

              # Self built
              snntorch
          ]))
        ];

        shellHook = ''
          echo "🐍 Python dev environment loaded!"
          python --version
        '';
      };
    };
}
