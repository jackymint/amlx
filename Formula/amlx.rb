class Amlx < Formula
  desc "MacBook-first local AI inference server with OpenAI-compatible API"
  homepage "https://github.com/jackymint/amlx"
  url "https://github.com/jackymint/amlx/archive/refs/tags/v1.1.6.tar.gz"
  sha256 "ff879b286636aa47a153e9d9f9ba74ec18f2eea8264c6dadafb87413fc1bb81c"
  license "Apache-2.0"
  version "1.1.6"

  depends_on "python@3.12"

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
