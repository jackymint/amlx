class Amlx < Formula
  desc "MacBook-first local AI inference server with OpenAI-compatible API"
  homepage "https://github.com/jackymint/amlx"
  url "https://github.com/jackymint/amlx/releases/download/v1.1.2/amlx-1.1.2-macos-arm64.tar.gz"
  sha256 "5e947679596630ebd391152f134c39d89da36b84e768854e696683b7d18462e3"
  license "Apache-2.0"
  version "1.1.2"

  def install
    bin.install "amlx"
  end

  test do
    assert_match version.to_s, shell_output("#{bin}/amlx version")
    system bin/"amlx", "--help"
  end
end
