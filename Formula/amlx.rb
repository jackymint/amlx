class Amlx < Formula
  desc "MacBook-first local AI inference server with OpenAI-compatible API"
  homepage "https://github.com/jackymint/amlx"
  url "https://github.com/jackymint/amlx/releases/download/v1.0.8/amlx-1.0.8-macos-arm64.tar.gz"
  sha256 "f6ed867a31ef3cdc46759cf4332a89cfc019c5bf67ca87e92e9a51de08b27f6c"
  license "Apache-2.0"
  version "1.0.8"

  def install
    bin.install "amlx"
  end

  test do
    assert_match version.to_s, shell_output("#{bin}/amlx version")
    system bin/"amlx", "--help"
  end
end
