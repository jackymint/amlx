class Amlx < Formula
  desc "MacBook-first local AI inference server with OpenAI-compatible API"
  homepage "https://github.com/jackymint/amlx"
  url "https://github.com/jackymint/amlx/releases/download/v1.0.9/amlx-1.0.9-macos-arm64.tar.gz"
  sha256 "53ad65f031016e66dff043a0a83eb4616c147827f47c8b5103664e667ac470d3"
  license "Apache-2.0"
  version "1.0.9"

  def install
    bin.install "amlx"
  end

  test do
    assert_match version.to_s, shell_output("#{bin}/amlx version")
    system bin/"amlx", "--help"
  end
end
