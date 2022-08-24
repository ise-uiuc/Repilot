from pathlib import Path
from realm.lsp import PipeLspAnalyzer
import json

java_server = 'java  \
        -Declipse.application=org.eclipse.jdt.ls.core.id1 \
        -Dosgi.bundles.defaultStartLevel=4 \
        -Declipse.product=org.eclipse.jdt.ls.core.product \
        -Dlog.level=ALL \
        -noverify \
        -Xmx1G \
        --add-modules=ALL-SYSTEM \
        --add-opens java.base/java.util=ALL-UNNAMED \
        --add-opens java.base/java.lang=ALL-UNNAMED \
        -jar /Users/nole/Developer/eclipse.jdt.ls/org.eclipse.jdt.ls.product/target/repository/plugins/org.eclipse.equinox.launcher_1.6.400.v20210924-0641.jar\
        -configuration /Users/nole/Developer/eclipse.jdt.ls/org.eclipse.jdt.ls.product/target/repository/config_mac \
        -data .path_to_data'

root = Path('/Users/nole/Developer/javasymbolsolver-maven-sample')
# root = Path('/tmp/lang_1_fixed')

file_path = root / 'src/main/java/com/yourorganization/maven_sample/MyAnalysis.java'
# file_path = root / 'src/main/java/org/apache/commons/lang3/math/NumberUtils.java'

analyzer = PipeLspAnalyzer(java_server)
analyzer.client.debug_call('initialize', {
    "processId": 710,
    "clientInfo": {
        "name": "Visual Studio Code",
        "version": "1.69.2"
    },
    "locale": "en-us",
    "rootPath": "/Users/nole/Developer/javasymbolsolver-maven-sample",
    "rootUri": "file:///Users/nole/Developer/javasymbolsolver-maven-sample",
    "capabilities": {
        "workspace": {
            "applyEdit": True,
            "workspaceEdit": {
                "documentChanges": True,
                "resourceOperations": [
                    "create",
                    "rename",
                    "delete"
                ],
                "failureHandling": "textOnlyTransactional",
                "normalizesLineEndings": True,
                "changeAnnotationSupport": {
                    "groupsOnLabel": True
                }
            },
            "didChangeConfiguration": {
                "dynamicRegistration": True
            },
            "didChangeWatchedFiles": {
                "dynamicRegistration": True
            },
            "symbol": {
                "dynamicRegistration": True,
                "symbolKind": {
                    "valueSet": [
                        1,
                        2,
                        3,
                        4,
                        5,
                        6,
                        7,
                        8,
                        9,
                        10,
                        11,
                        12,
                        13,
                        14,
                        15,
                        16,
                        17,
                        18,
                        19,
                        20,
                        21,
                        22,
                        23,
                        24,
                        25,
                        26
                    ]
                },
                "tagSupport": {
                    "valueSet": [
                        1
                    ]
                }
            },
            "codeLens": {
                "refreshSupport": True
            },
            "executeCommand": {
                "dynamicRegistration": True
            },
            "configuration": True,
            "workspaceFolders": True,
            "semanticTokens": {
                "refreshSupport": True
            },
            "fileOperations": {
                "dynamicRegistration": True,
                "didCreate": True,
                "didRename": True,
                "didDelete": True,
                "willCreate": True,
                "willRename": True,
                "willDelete": True
            }
        },
        "textDocument": {
            "publishDiagnostics": {
                "relatedInformation": True,
                "versionSupport": False,
                "tagSupport": {
                    "valueSet": [
                        1,
                        2
                    ]
                },
                "codeDescriptionSupport": True,
                "dataSupport": True
            },
            "synchronization": {
                "dynamicRegistration": True,
                "willSave": True,
                "willSaveWaitUntil": True,
                "didSave": True
            },
            "completion": {
                "dynamicRegistration": True,
                "contextSupport": True,
                "completionItem": {
                    "snippetSupport": True,
                    "commitCharactersSupport": True,
                    "documentationFormat": [
                        "markdown",
                        "plaintext"
                    ],
                    "deprecatedSupport": True,
                    "preselectSupport": True,
                    "tagSupport": {
                        "valueSet": [
                            1
                        ]
                    },
                    "insertReplaceSupport": True,
                    "resolveSupport": {
                        "properties": [
                            "documentation",
                            "detail",
                            "additionalTextEdits"
                        ]
                    },
                    "insertTextModeSupport": {
                        "valueSet": [
                            1,
                            2
                        ]
                    },
                    "labelDetailsSupport": True
                },
                "insertTextMode": 2,
                "completionItemKind": {
                    "valueSet": [
                        1,
                        2,
                        3,
                        4,
                        5,
                        6,
                        7,
                        8,
                        9,
                        10,
                        11,
                        12,
                        13,
                        14,
                        15,
                        16,
                        17,
                        18,
                        19,
                        20,
                        21,
                        22,
                        23,
                        24,
                        25
                    ]
                }
            },
            "hover": {
                "dynamicRegistration": True,
                "contentFormat": [
                    "markdown",
                    "plaintext"
                ]
            },
            "signatureHelp": {
                "dynamicRegistration": True,
                "signatureInformation": {
                    "documentationFormat": [
                        "markdown",
                        "plaintext"
                    ],
                    "parameterInformation": {
                        "labelOffsetSupport": True
                    },
                    "activeParameterSupport": True
                },
                "contextSupport": True
            },
            "definition": {
                "dynamicRegistration": True,
                "linkSupport": True
            },
            "references": {
                "dynamicRegistration": True
            },
            "documentHighlight": {
                "dynamicRegistration": True
            },
            "documentSymbol": {
                "dynamicRegistration": True,
                "symbolKind": {
                    "valueSet": [
                        1,
                        2,
                        3,
                        4,
                        5,
                        6,
                        7,
                        8,
                        9,
                        10,
                        11,
                        12,
                        13,
                        14,
                        15,
                        16,
                        17,
                        18,
                        19,
                        20,
                        21,
                        22,
                        23,
                        24,
                        25,
                        26
                    ]
                },
                "hierarchicalDocumentSymbolSupport": True,
                "tagSupport": {
                    "valueSet": [
                        1
                    ]
                },
                "labelSupport": True
            },
            "codeAction": {
                "dynamicRegistration": True,
                "isPreferredSupport": True,
                "disabledSupport": True,
                "dataSupport": True,
                "resolveSupport": {
                    "properties": [
                        "edit"
                    ]
                },
                "codeActionLiteralSupport": {
                    "codeActionKind": {
                        "valueSet": [
                            "",
                            "quickfix",
                            "refactor",
                            "refactor.extract",
                            "refactor.inline",
                            "refactor.rewrite",
                            "source",
                            "source.organizeImports"
                        ]
                    }
                },
                "honorsChangeAnnotations": False
            },
            "codeLens": {
                "dynamicRegistration": True
            },
            "formatting": {
                "dynamicRegistration": True
            },
            "rangeFormatting": {
                "dynamicRegistration": True
            },
            "onTypeFormatting": {
                "dynamicRegistration": True
            },
            "rename": {
                "dynamicRegistration": True,
                "prepareSupport": True,
                "prepareSupportDefaultBehavior": 1,
                "honorsChangeAnnotations": True
            },
            "documentLink": {
                "dynamicRegistration": True,
                "tooltipSupport": True
            },
            "typeDefinition": {
                "dynamicRegistration": True,
                "linkSupport": True
            },
            "implementation": {
                "dynamicRegistration": True,
                "linkSupport": True
            },
            "colorProvider": {
                "dynamicRegistration": True
            },
            "foldingRange": {
                "dynamicRegistration": True,
                "rangeLimit": 5000,
                "lineFoldingOnly": True
            },
            "declaration": {
                "dynamicRegistration": True,
                "linkSupport": True
            },
            "selectionRange": {
                "dynamicRegistration": True
            },
            "callHierarchy": {
                "dynamicRegistration": True
            },
            "semanticTokens": {
                "dynamicRegistration": True,
                "tokenTypes": [
                    "namespace",
                    "type",
                    "class",
                    "enum",
                    "interface",
                    "struct",
                    "typeParameter",
                    "parameter",
                    "variable",
                    "property",
                    "enumMember",
                    "event",
                    "function",
                    "method",
                    "macro",
                    "keyword",
                    "modifier",
                    "comment",
                    "string",
                    "number",
                    "regexp",
                    "operator"
                ],
                "tokenModifiers": [
                    "declaration",
                    "definition",
                    "readonly",
                    "static",
                    "deprecated",
                    "abstract",
                    "async",
                    "modification",
                    "documentation",
                    "defaultLibrary"
                ],
                "formats": [
                    "relative"
                ],
                "requests": {
                    "range": True,
                    "full": {
                        "delta": True
                    }
                },
                "multilineTokenSupport": False,
                "overlappingTokenSupport": False
            },
            "linkedEditingRange": {
                "dynamicRegistration": True
            }
        },
        "window": {
            "showMessage": {
                "messageActionItem": {
                    "additionalPropertiesSupport": True
                }
            },
            "showDocument": {
                "support": True
            },
            "workDoneProgress": True
        },
        "general": {
            "staleRequestSupport": {
                "cancel": True,
                "retryOnContentModified": [
                    "textDocument/semanticTokens/full",
                    "textDocument/semanticTokens/range",
                    "textDocument/semanticTokens/full/delta"
                ]
            },
            "regularExpressions": {
                "engine": "ECMAScript",
                "version": "ES2020"
            },
            "markdown": {
                "parser": "marked",
                "version": "1.1.0"
            }
        }
    },
    "initializationOptions": {
        "bundles": [],
        "workspaceFolders": [
            "file:///Users/nole/Developer/javasymbolsolver-maven-sample"
        ],
        "settings": {
            "java": {
                "home": "/Users/nole/Library/Caches/Coursier/arc/https/github.com/adoptium/temurin18-binaries/releases/download/jdk-18.0.1%252B10/OpenJDK18U-jdk_x64_mac_hotspot_18.0.1_10.tar.gz/jdk-18.0.1+10/Contents/Home",
                "jdt": {
                    "ls": {
                        "java": {
                            "home": None,
                        },
                        "vmargs": "-XX:+UseParallelGC -XX:GCTimeRatio=4 -XX:AdaptiveSizePolicyWeight=90 -Dsun.zip.disableMemoryMapping=True -Xmx1G -Xms100m",
                        "lombokSupport": {
                            "enabled": True
                        }
                    }
                },
                "errors": {
                    "incompleteClasspath": {
                        "severity": "warning"
                    }
                },
                "configuration": {
                    "checkProjectSettingsExclusions": False,
                    "updateBuildConfiguration": "interactive",
                    "maven": {
                        "userSettings": None,
                        "globalSettings": None,
                        "notCoveredPluginExecutionSeverity": "warning"
                    },
                    "workspaceCacheLimit": 90,
                    "runtimes": []
                },
                "trace": {
                    "server": "verbose"
                },
                "import": {
                    "maven": {
                        "enabled": True
                    },
                    "gradle": {
                        "enabled": True,
                        "wrapper": {
                            "enabled": True
                        },
                        "version": None,
                        "home": None,
                        "java": {
                            "home": None,
                        },
                        "offline": {
                            "enabled": False
                        },
                        "arguments": None,
                        "jvmArguments": None,
                        "user": {
                            "home": None,
                        }
                    },
                    "exclusions": [
                        "**/node_modules/**",
                        "**/.metadata/**",
                        "**/archetype-resources/**",
                        "**/META-INF/maven/**"
                    ],
                    "generatesMetadataFilesAtProjectRoot": False
                },
                "maven": {
                    "downloadSources": False,
                    "updateSnapshots": False
                },
                "eclipse": {
                    "downloadSources": False
                },
                "referencesCodeLens": {
                    "enabled": False
                },
                "signatureHelp": {
                    "enabled": False,
                    "description": {
                        "enabled": False
                    }
                },
                "implementationsCodeLens": {
                    "enabled": False
                },
                "format": {
                    "enabled": True,
                    "settings": {
                        "url": None,
                        "profile": None,
                    },
                    "comments": {
                        "enabled": True
                    },
                    "onType": {
                        "enabled": True
                    },
                    "insertSpaces": True,
                    "tabSize": 4
                },
                "saveActions": {
                    "organizeImports": False
                },
                "project": {
                    "referencedLibraries": [
                        "lib/**/*.jar"
                    ],
                    "importOnFirstTimeStartup": "automatic",
                    "importHint": True,
                    "resourceFilters": [
                        "node_modules",
                        ".git"
                    ],
                    "encoding": "ignore"
                },
                "contentProvider": {
                    "preferred": None,
                },
                "autobuild": {
                    "enabled": True
                },
                "maxConcurrentBuilds": 1,
                "recommendations": {
                    "dependency": {
                        "analytics": {
                            "show": True
                        }
                    }
                },
                "completion": {
                    "maxResults": 0,
                    "enabled": True,
                    "guessMethodArguments": False,
                    "favoriteStaticMembers": [
                        "org.junit.Assert.*",
                        "org.junit.Assume.*",
                        "org.junit.jupiter.api.Assertions.*",
                        "org.junit.jupiter.api.Assumptions.*",
                        "org.junit.jupiter.api.DynamicContainer.*",
                        "org.junit.jupiter.api.DynamicTest.*",
                        "org.mockito.Mockito.*",
                        "org.mockito.ArgumentMatchers.*",
                        "org.mockito.Answers.*"
                    ],
                    "filteredTypes": [
                        "java.awt.*",
                        "com.sun.*",
                        "sun.*",
                        "jdk.*",
                        "org.graalvm.*",
                        "io.micrometer.shaded.*"
                    ],
                    "importOrder": [
                        "java",
                        "javax",
                        "org",
                        "com"
                    ]
                },
                "foldingRange": {
                    "enabled": True
                },
                "progressReports": {
                    "enabled": True
                },
                "codeGeneration": {
                    "hashCodeEquals": {
                        "useJava7Objects": False,
                        "useInstanceof": False
                    },
                    "useBlocks": False,
                    "generateComments": False,
                    "toString": {
                        "template": "${object.className} [${member.name()}=${member.value}, ${otherMembers}]",
                        "codeStyle": "STRING_CONCATENATION",
                        "SkipNone,Values": False,
                        "listArrayContents": True,
                        "limitElements": 0
                    },
                    "insertionLocation": "afterCursor"
                },
                "selectionRange": {
                    "enabled": True
                },
                "showBuildStatusOnStart": {
                    "enabled": "notification"
                },
                "server": {
                    "launchMode": "Hybrid"
                },
                "sources": {
                    "organizeImports": {
                        "starThreshold": 99,
                        "staticStarThreshold": 99
                    }
                },
                "imports": {
                    "gradle": {
                        "wrapper": {
                            "checksums": []
                        }
                    }
                },
                "templates": {
                    "fileHeader": [],
                    "typeComment": []
                },
                "references": {
                    "includeAccessors": True,
                    "includeDecompiledSources": True
                },
                "typeHierarchy": {
                    "lazyLoad": False
                },
                "settings": {
                    "url": None,
                },
                "symbols": {
                    "includeSourceMethodDeclarations": False
                },
                "quickfix": {
                    "showAt": "line"
                },
                "inlayHints": {
                    "parameterNames": {
                        "enabled": "literals",
                        "exclusions": []
                    }
                }
            }
        },
        "extendedClientCapabilities": {
            "progressReportProvider": True,
            "classFileContentsSupport": True,
            "overrideMethodsPromptSupport": True,
            "hashCodeEqualsPromptSupport": True,
            "advancedOrganizeImportsSupport": True,
            "generateToStringPromptSupport": True,
            "advancedGenerateAccessorsSupport": True,
            "generateConstructorsPromptSupport": True,
            "generateDelegateMethodsPromptSupport": True,
            "advancedExtractRefactoringSupport": True,
            "inferSelectionSupport": [
                "extractMethod",
                "extractVariable",
                "extractField"
            ],
            "moveRefactoringSupport": True,
            "clientHoverProvider": True,
            "clientDocumentSymbolProvider": True,
            "gradleChecksumWrapperPromptSupport": True,
            "resolveAdditionalTextEditsSupport": True,
            "advancedIntroduceParameterRefactoringSupport": True,
            "actionableRuntimeNotificationSupport": True,
            "shouldLanguageServerExitOnShutdown": True,
            "onCompletionItemSelectedCommand": "editor.action.triggerParameterHints"
        },
        "triggerFiles": [
            "file:///Users/nole/Developer/javasymbolsolver-maven-sample/src/main/java/com/yourorganization/maven_sample/MyAnalysis.java"
        ]
    },
    "trace": "verbose",
    "workspaceFolders": [
        {
            "uri": "file:///Users/nole/Developer/javasymbolsolver-maven-sample",
            "name": "javasymbolsolver-maven-sample"
        }
    ]
})
analyzer.client._initialized()
# analyzer.init(root)
analyzer.client._did_open('java', file_path)
# init = analyzer.init(root)
# print(init['result']['capabilities'])
# print(analyzer._register('workspace/symbol'))
# # analyzer._did_open('java', file_path)

completions = analyzer.complete(file_path, 18, 13)
print(completions)

# print(analyzer._doc_symbol(file_path)['result'])
# print(analyzer.client._wkspace_symbol('test'))
print([x['name'] for x in analyzer.client._doc_symbol(file_path)['result']])

analyzer.client._did_save(file_path)
# print(analyzer.client._diagnostic(file_path))
