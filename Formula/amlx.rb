class Amlx < Formula
  desc "MacBook-first local AI inference server with OpenAI-compatible API"
  homepage "https://github.com/jackymint/amlx"
  url "https://github.com/jackymint/amlx/releases/download/v1.0.1/amlx-1.0.1-macos-arm64.tar.gz"
  sha256 "f4cf1406a4a52f42fce424574de87bd48e83bac799f3dcc1d82a144b225c1fd1"
  license "Apache-2.0"
  version "1.0.1"

  def install
    bin.install "amlx"
  end

  test do
    assert_match version.to_s, shell_output("#{bin}/amlx version")
    system bin/"amlx", "--help"
  end
end
