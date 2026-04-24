class Amlx < Formula
  include Language::Python::Virtualenv

  desc "MacBook-first local AI inference server with OpenAI-compatible API"
  homepage "https://github.com/jackymint/amlx"
  url "https://github.com/jackymint/amlx/archive/refs/tags/v0.1.0.tar.gz"
  sha256 "TODO: replace with sha256 of release tarball (brew fetch --build-from-source amlx)"
  license "Apache-2.0"

  bottle :unneeded

  depends_on "python@3.12"
  depends_on :macos # requires Apple Silicon for MLX backend

  # --- core dependencies ---

  resource "annotated-types" do
    url "https://files.pythonhosted.org/packages/source/a/annotated_types/annotated_types-0.7.0.tar.gz"
    sha256 "aff07c09a53a08bc8cfccb9c85b05f1aa9a2a6f23728d790723543408344ce89"
  end

  resource "anyio" do
    url "https://files.pythonhosted.org/packages/source/a/anyio/anyio-4.9.0.tar.gz"
    sha256 "673c0c244e15788651a4ff38710fea9675823028a6f08a5eda361a487542a3ae"
  end

  resource "certifi" do
    url "https://files.pythonhosted.org/packages/source/c/certifi/certifi-2025.1.31.tar.gz"
    sha256 "3d5da6f9b13f3c50df65c62bf24fca1d6dcf2dab2e7f7f2d52e67c3ef3bbf29"
  end

  resource "click" do
    url "https://files.pythonhosted.org/packages/source/c/click/click-8.1.8.tar.gz"
    sha256 "ed53c9d8990d83c2a27deae68e4ee337473f6330c040a31d4225c9574d16096a"
  end

  resource "fastapi" do
    url "https://files.pythonhosted.org/packages/source/f/fastapi/fastapi-0.115.12.tar.gz"
    sha256 "1e2c2a2646272ab1b5f8d896b1c0b1a8a6775a4b1bf22cd9b3a45b37aec960c"
  end

  resource "filelock" do
    url "https://files.pythonhosted.org/packages/source/f/filelock/filelock-3.18.0.tar.gz"
    sha256 "adbc88eabb99d2fec8c9c1b229963d8e6393ba12a3b28eb6b366ac2bed8b149a"
  end

  resource "fsspec" do
    url "https://files.pythonhosted.org/packages/source/f/fsspec/fsspec-2025.3.2.tar.gz"
    sha256 "3e5924f9b49f1afe78d4dc5ecfc3b36ae94d41609f41b82094cf78e1a4b6cfbb"
  end

  resource "h11" do
    url "https://files.pythonhosted.org/packages/source/h/h11/h11-0.14.0.tar.gz"
    sha256 "8f19fbbe99e72420ff35c00b27a34cb9937e902a8b810e2c88300c9f0a1178e6"
  end

  resource "httptools" do
    url "https://files.pythonhosted.org/packages/source/h/httptools/httptools-0.6.4.tar.gz"
    sha256 "4e93eee4add6493b59a5c514da98c939b244fce4a0d8879cd3f466562f4b7d5c"
  end

  resource "huggingface-hub" do
    url "https://files.pythonhosted.org/packages/source/h/huggingface_hub/huggingface_hub-0.30.2.tar.gz"
    sha256 "73a2b7b4be32c3fb2e2e0c5b1f13dc440ab2428f09c57dab9c3b540dce892826"
  end

  resource "idna" do
    url "https://files.pythonhosted.org/packages/source/i/idna/idna-3.10.tar.gz"
    sha256 "12f65c9b470abda6dc35cf5e1d1317a3a7fd6328ab7a301a09931cb553e8d7a0"
  end

  resource "packaging" do
    url "https://files.pythonhosted.org/packages/source/p/packaging/packaging-24.2.tar.gz"
    sha256 "c228a6dc5e932d346bc5739379109d49e8853dd8223571c7c5b55260edc0b97f"
  end

  resource "pydantic" do
    url "https://files.pythonhosted.org/packages/source/p/pydantic/pydantic-2.11.3.tar.gz"
    sha256 "7471657138c16adad9322fe3070c0116dd6c3ad8d649300e3cbdfe91f4db4ec3"
  end

  resource "pydantic-core" do
    url "https://files.pythonhosted.org/packages/source/p/pydantic_core/pydantic_core-2.33.1.tar.gz"
    sha256 "bcc9c6fdb0ced789ad08b640f158b456580e1f01e3ae916a18c78a3f1e25a2c6"
  end

  resource "python-dotenv" do
    url "https://files.pythonhosted.org/packages/source/p/python-dotenv/python_dotenv-1.1.0.tar.gz"
    sha256 "41f90bc6f5f177fb41f53e87666db362025010eb28f6076b8bfff8b0d965e0ae"
  end

  resource "pyyaml" do
    url "https://files.pythonhosted.org/packages/source/P/PyYAML/PyYAML-6.0.2.tar.gz"
    sha256 "d584d9ec91ad65861cc08d42e834324ef890a082e591037abe114850ff7bbc3e"
  end

  resource "requests" do
    url "https://files.pythonhosted.org/packages/source/r/requests/requests-2.32.3.tar.gz"
    sha256 "55365417734eb18255590a9f9f658b8027b0e6b7f8d17db0abc07b7d3d4a9a7b"
  end

  resource "rich" do
    url "https://files.pythonhosted.org/packages/source/r/rich/rich-14.0.0.tar.gz"
    sha256 "82b1d18cdb33e202d2e7cf4ec9021f7f6b316fde31b9afd3b943ebb22cee4b9e"
  end

  resource "shellingham" do
    url "https://files.pythonhosted.org/packages/source/s/shellingham/shellingham-1.5.4.tar.gz"
    sha256 "8dbca0739d487e5bd35ab3ca4b36e11c4078f3a234bfce294b0a0291363404de"
  end

  resource "sniffio" do
    url "https://files.pythonhosted.org/packages/source/s/sniffio/sniffio-1.3.1.tar.gz"
    sha256 "f4324edc670a0f49750a81b895f35c3a939d4a1a6eedbc2dd11edd7b58f0b54d"
  end

  resource "starlette" do
    url "https://files.pythonhosted.org/packages/source/s/starlette/starlette-0.46.1.tar.gz"
    sha256 "e0d6d33c6bea2c7d8e8e1e3c3b9b16b4c0a14b04eabfda9c01aae5c7a8b1f3f2"
  end

  resource "tqdm" do
    url "https://files.pythonhosted.org/packages/source/t/tqdm/tqdm-4.67.1.tar.gz"
    sha256 "f8aef9c52c08c13a65f30ea34f4e5aac3fd1a34959879d7e59e63027286a87d2"
  end

  resource "typer" do
    url "https://files.pythonhosted.org/packages/source/t/typer/typer-0.15.2.tar.gz"
    sha256 "b9e5b7f7af80c6b4e2f4b4b23ec1c0ab5df9f6fa6c8ec41c99326a3df3e5a0c5"
  end

  resource "typing-extensions" do
    url "https://files.pythonhosted.org/packages/source/t/typing_extensions/typing_extensions-4.13.0.tar.gz"
    sha256 "0a4ac8b73b37b746c47e30c6afc60b74eedbab5d9a2f53c0c91b2481bc46b9e4"
  end

  resource "urllib3" do
    url "https://files.pythonhosted.org/packages/source/u/urllib3/urllib3-2.4.0.tar.gz"
    sha256 "414bc6535b787febd7567804cc015feb5c23afe74e5d5e7489a8b00ceea07d81"
  end

  resource "uvicorn" do
    url "https://files.pythonhosted.org/packages/source/u/uvicorn/uvicorn-0.34.0.tar.gz"
    sha256 "404051de8069bd36c64e5374cf38edb8d5d7740d9af17ce3aaa7df0afce71bb7"
  end

  resource "uvloop" do
    url "https://files.pythonhosted.org/packages/source/u/uvloop/uvloop-0.21.0.tar.gz"
    sha256 "3bf12b0fda68447806a7ad847bfa591613177275d35b6724b1ee573faa3704e3"
  end

  resource "watchfiles" do
    url "https://files.pythonhosted.org/packages/source/w/watchfiles/watchfiles-1.0.4.tar.gz"
    sha256 "9f9a0f36e0e00e5e4dccc6ce0d01d7db3f7c66e4e3d6ddc53ac57d8ab89e0e00"
  end

  resource "websockets" do
    url "https://files.pythonhosted.org/packages/source/w/websockets/websockets-15.0.1.tar.gz"
    sha256 "82544de02076bafba038ce055ee6412d68da13ab47f0c60f9291a71ffe4b0a5a"
  end

  # --- Apple Silicon MLX backend ---

  resource "mlx" do
    url "https://files.pythonhosted.org/packages/source/m/mlx/mlx-0.22.0.tar.gz"
    sha256 "TODO: replace with actual sha256 from PyPI"
  end

  resource "mlx-lm" do
    url "https://files.pythonhosted.org/packages/source/m/mlx_lm/mlx_lm-0.22.0.tar.gz"
    sha256 "TODO: replace with actual sha256 from PyPI"
  end

  def install
    virtualenv_install_with_resources
  end

  test do
    assert_match version.to_s, shell_output("#{bin}/amlx version")
    system bin/"amlx", "--help"
  end
end
