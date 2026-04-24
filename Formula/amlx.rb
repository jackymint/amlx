class Amlx < Formula
  desc "MacBook-first local AI inference server with OpenAI-compatible API"
  homepage "https://github.com/jackymint/amlx"
  url "https://github.com/jackymint/amlx/releases/download/v1.0.5/amlx-1.0.5-macos-arm64.tar.gz"
  sha256 "6543a683cfd5b5c08e193462a514c70e83a400c2616e52d905b76df42e55912f"
  license "Apache-2.0"
  version "1.0.5"

  def install
    bin.install "amlx"
  end

  test do
    assert_match version.to_s, shell_output("#{bin}/amlx version")
    system bin/"amlx", "--help"
  end
end
