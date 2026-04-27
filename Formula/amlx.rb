class Amlx < Formula
  desc "MacBook-first local AI inference server with OpenAI-compatible API"
  homepage "https://github.com/jackymint/amlx"
  url "https://github.com/jackymint/amlx/releases/download/v1.1.3/amlx-1.1.3-macos-arm64.tar.gz"
  sha256 "6eb1c2c3561921ab3bfde5072266bd14910ec9d2b0dd1c335b777d91840bf118"
  license "Apache-2.0"
  version "1.1.3"

  def install
    bin.install "amlx"
  end

  test do
    assert_match version.to_s, shell_output("#{bin}/amlx version")
    system bin/"amlx", "--help"
  end
end
