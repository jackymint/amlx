class Amlx < Formula
  desc "MacBook-first local AI inference server with OpenAI-compatible API"
  homepage "https://github.com/jackymint/amlx"
  url "https://github.com/jackymint/amlx/releases/download/v1.1.4/amlx-1.1.4-macos-arm64.tar.gz"
  sha256 "283040ea1a52ea4b45a1facea66ad6abd2ed172969dbdd7499a7720e59d2acbf"
  license "Apache-2.0"
  version "1.1.4"

  def install
    bin.install "amlx"
  end

  test do
    assert_match version.to_s, shell_output("#{bin}/amlx version")
    system bin/"amlx", "--help"
  end
end
