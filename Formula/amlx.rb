class Amlx < Formula
  desc "MacBook-first local AI inference server with OpenAI-compatible API"
  homepage "https://github.com/jackymint/amlx"
  url "https://github.com/jackymint/amlx/releases/download/v1.1.5/amlx-1.1.5-macos-arm64.tar.gz"
  sha256 "d23248f2bd22d96db52c5bd557b6148bfcda7e2fb30dc80b9ea2cb2f0586f29d"
  license "Apache-2.0"
  version "1.1.5"

  def install
    bin.install "amlx"
  end

  test do
    assert_match version.to_s, shell_output("#{bin}/amlx version")
    system bin/"amlx", "--help"
  end
end
