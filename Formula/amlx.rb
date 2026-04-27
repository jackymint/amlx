class Amlx < Formula
  desc "MacBook-first local AI inference server with OpenAI-compatible API"
  homepage "https://github.com/jackymint/amlx"
  url "https://github.com/jackymint/amlx/archive/refs/tags/v1.1.7.tar.gz"
  sha256 "3e972795698b605f8f2230d68373731257594aa615aa6c4766d52050370e0cb3"
  license "Apache-2.0"
  version "1.1.7"

  depends_on "python@3.12"

  # Prevent Homebrew from relinking .so files inside the venv (pydantic_core has no header space)
  skip_clean "libexec"

  def install
    python = Formula["python@3.12"].opt_bin/"python3.12"
    system python, "-m", "venv", libexec
    system libexec/"bin/pip", "install", "--only-binary=:all:", ".[mlx]"
    bin.install_symlink libexec/"bin/amlx"
  end

  test do
    assert_match version.to_s, shell_output("#{bin}/amlx version")
    system bin/"amlx", "--help"
  end
end
